import os
import sncosmo
import requests
import numpy as np
from numpy.lib import recfunctions
from pathlib import Path
import shutil

from sqlalchemy.orm import deferred


import sqlalchemy as sa
import time
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy import Index
from sqlalchemy import func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property

from skyportal import models
from skyportal.models import DBSession
from skyportal.model_util import create_tables, drop_tables

from datetime import datetime

from secrets import get_secret
from photometry import aperture_photometry, APER_KEY
from swarp import ensure_images_have_the_same_properties, run_coadd, run_align
import archive
from hotpants import prepare_hotpants
from filterobjects import filter_sexcat, BRAAI_MODEL
from seeing import estimate_seeing

import sextractor

import requests

import fitsio
import subprocess
import uuid
import warnings
import pandas as pd

import photutils
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy import convolution
from astropy.coordinates import SkyCoord

from matplotlib import colors
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Ellipse
import publish

from crossmatch import xmatch

BIG_RMS = np.sqrt(50000.)
BKG_BOX_SIZE = 128
DETECT_NSIGMA = 1.5
DETECT_NPIX = 5
TABLE_COLUMNS = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                 'sky_centroid_icrs', 'source_sum', 'source_sum_err',
                 'orientation', 'eccentricity', 'semimajor_axis_sigma',
                 'semiminor_axis_sigma']


SEXTRACTOR_EQUIVALENTS = ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE', 'X_WORLD',
                          'Y_WORLD', 'FLUX_APER', 'FLUXERR_APER',
                          'THETA_WORLD', 'ELLIPTICITY', 'A_IMAGE', 'B_IMAGE']
MJD_TO_JD = 2400000.5
CMAP_RANDOM_SEED = 8675309

NERSC_PREFIX = '/global/cfs/cdirs/m937/www/data'
URL_PREFIX = 'https://portal.nersc.gov/cfs/m937'
STAMP_PREFIX = '/global/cfs/cdirs/m937/www'

#NERSC_PREFIX = '/global/cfs/projectdirs/ptf/www/ztf/data'
#URL_PREFIX = 'https://portal.nersc.gov/cfs/ptf/ztf'
#STAMP_PREFIX = '/global/cfs/projectdirs/ptf/www/ztf'

#NERSC_PREFIX = '/global/projecta/projectdirs/tmp_astro250/data'
#URL_PREFIX = 'https://portal.nersc.gov/project/astro250/ro_workaround'
#STAMP_PREFIX = '/global/projecta/projectdirs/tmp_astro250'

GROUP_PROPERTIES = ['field', 'ccdid', 'qid', 'fid']
MATCH_RADIUS_DEG = 0.0002777 * 2.0
N_PREV_SINGLE = 1
N_PREV_MULTI = 1
RB_ASSOC_MIN = 0.2

MASK_BITS = {
    'BIT00': 0,
    'BIT01': 1,
    'BIT02': 2,
    'BIT03': 3,
    'BIT04': 4,
    'BIT05': 5,
    'BIT06': 6,
    'BIT07': 7,
    'BIT08': 8,
    'BIT09': 9,
    'BIT10': 10,
    'BIT11': 11,
    'BIT12': 12,
    'BIT13': 13,
    'BIT14': 14,
    'BIT15': 15,
    'BIT16': 16
}

BAD_BITS = np.asarray([0, 2, 3, 4, 5, 7, 8, 9, 10, 16, 17])
BAD_SUM = int(np.sum(2 ** BAD_BITS))

MASK_COMMENTS = {
    'BIT00': 'AIRCRAFT/SATELLITE TRACK',
    'BIT01': 'CONTAINS SEXTRACTOR DETECTION',
    'BIT02': 'LOW RESPONSIVITY',
    'BIT03': 'HIGH RESPONSIVITY',
    'BIT04': 'NOISY',
    'BIT05': 'GHOST FROM BRIGHT SOURCE',
    'BIT06': 'RESERVED FOR FUTURE USE',
    'BIT07': 'PIXEL SPIKE (POSSIBLE RAD HIT)',
    'BIT08': 'SATURATED',
    'BIT09': 'DEAD (UNRESPONSIVE)',
    'BIT10': 'NAN (not a number)',
    'BIT11': 'CONTAINS PSF-EXTRACTED SOURCE POSITION',
    'BIT12': 'HALO FROM BRIGHT SOURCE',
    'BIT13': 'RESERVED FOR FUTURE USE',
    'BIT14': 'RESERVED FOR FUTURE USE',
    'BIT15': 'RESERVED FOR FUTURE USE',
    'BIT16': 'NON-DATA SECTION FROM SWARP ALIGNMENT'
}

fid_map = {
    1: 'zg',
    2: 'zr',
    3: 'zi'
}


def plot_triplet(tr):
    fig = plt.figure(figsize=(8, 2), dpi=100)
    ax = fig.add_subplot(131)
    ax.axis('off')
    ax.imshow(tr[:, :, 0], origin='upper', cmap=plt.cm.bone)
    ax2 = fig.add_subplot(132)
    ax2.axis('off')
    ax2.imshow(tr[:, :, 1], origin='upper', cmap=plt.cm.bone)
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.imshow(tr[:, :, 2], origin='upper', cmap=plt.cm.bone)
    plt.show()


def discrete_cmap(ncolors):
    """Create a ListedColorMap with `ncolors` randomly-generated colors
    that can be used to color an IntegerFITSImage.

    The first color in the list is always black."""

    prng = np.random.RandomState(CMAP_RANDOM_SEED)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))
    rgb[0] = (0.,) * 3
    return colors.ListedColormap(rgb)


def show_images(image_or_images, catalog=None, titles=None, reproject=False,
                ds9=False):
    imgs = np.atleast_1d(image_or_images)
    n = len(imgs)

    if ds9:
        if catalog is not None:
            reg = PipelineRegionFile.from_catalog(catalog)
        cmd = '%ds9 -zscale '
        for img in imgs:
            img.save()
            cmd += f' {img.local_path}'
            if catalog is not None:
                cmd += f' -region {reg.local_path}'
        cmd += ' -lock frame wcs'
        print(cmd)
    else:

        if titles is not None and len(titles) != n:
            raise ValueError('len(titles) != len(images)')

        ncols = min(n, 3)
        nrows = (n - 1) // 3 + 1

        align_target = imgs[0]
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True,
                               sharey=True,
                               subplot_kw={
                                   'projection': WCS(
                                       align_target.astropy_header)
                               } if reproject else None)

        ax = np.atleast_1d(ax)
        for a in ax.ravel()[n:]:
            a.set_visible(False)


        for i, (im, a) in enumerate(zip(imgs, ax.ravel())):
            im.show(a, align_to=align_target if reproject else None)
            if catalog is not None:
                filtered = 'GOODCUT' in catalog.data.dtype.names
                my_xy = im.wcs.all_world2pix(list(zip(catalog.data[
                                                          'X_WORLD'],
                                                      catalog.data[
                                                          'Y_WORLD'
                                                      ])), 0)
                for row, xy in zip(catalog.data, my_xy):

                    e = Ellipse(xy=xy,
                                width=6 * row['A_IMAGE'],
                                height=6 * row['B_IMAGE'],
                                angle=row['THETA_IMAGE'] * 180. / np.pi)
                    e.set_facecolor('none')

                    if filtered and row['GOODCUT'] == 1:
                        e.set_edgecolor('lime')
                    else:
                        e.set_edgecolor('red')
                    a.add_artist(e)

            if titles is not None:
                a.set_title(titles[i])

        fig.tight_layout()
        fig.show()
        return fig


def sub_name(frame, template):
    frame = f'{frame}'
    template = f'{template}'

    refp = os.path.basename(template)[:-5]
    newp = os.path.basename(frame)[:-5]

    outdir = os.path.dirname(frame)

    subp = '_'.join([newp, refp])

    sub = os.path.join(outdir, 'sub.%s.fits' % subp)
    return sub


def init_db(old=False, timeout=None):
    hpss_dbhost = get_secret('hpss_dbhost')
    hpss_dbport = get_secret('hpss_dbport')
    hpss_dbusername = get_secret('hpss_dbusername')
    hpss_dbname = get_secret('hpss_dbname') if not old else get_secret('olddb')
    hpss_dbpassword = get_secret('hpss_dbpassword')

    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(hpss_dbusername, hpss_dbpassword or '',
                     hpss_dbhost or '',
                     hpss_dbport or '',
                     hpss_dbname)


    kwargs = {}
    if timeout is not None:
        kwargs['connect_args'] = {"options": f"-c statement_timeout={timeout}"}

    conn = sa.create_engine(url, client_encoding='utf8', **kwargs)

    DBSession.configure(bind=conn)
    models.Base.metadata.bind = conn

    return conn


def model_representation(o):
    if sa.inspection.inspect(o).expired:
        DBSession().refresh(o)
    inst = sa.inspect(o)
    attr_list = [f"{g.key}={getattr(o, g.key)}"
                 for g in inst.mapper.column_attrs]
    return f"<{type(o).__name__}({', '.join(attr_list)})>"


models.Base.__repr__ = model_representation
models.Base.modified = sa.Column(
    sa.DateTime(timezone=False),
    default=sa.func.now(),
    onupdate=sa.func.now()
)


def join_model(join_table, model_1, model_2, column_1=None, column_2=None,
               fk_1='id', fk_2='id', base=models.Base):
    """Helper function to create a join table for a many-to-many relationship.

    Parameters
    ----------
    join_table : str
        Name of the new table to be created.
    model_1 : str
        First model in the relationship.
    model_2 : str
        Second model in the relationship.
    column_1 : str, optional
        Name of the join table column corresponding to `model_1`. If `None`,
        then {`table1`[:-1]_id} will be used (e.g., `user_id` for `users`).
    column_2 : str, optional
        Name of the join table column corresponding to `model_2`. If `None`,
        then {`table2`[:-1]_id} will be used (e.g., `user_id` for `users`).
    fk_1 : str, optional
        Name of the column from `model_1` that the foreign key should refer to.
    fk_2 : str, optional
        Name of the column from `model_2` that the foreign key should refer to.
    base : sqlalchemy.ext.declarative.api.DeclarativeMeta
        SQLAlchemy model base to subclass.

    Returns
    -------
    sqlalchemy.ext.declarative.api.DeclarativeMeta
        SQLAlchemy association model class
    """
    table_1 = model_1.__tablename__
    table_2 = model_2.__tablename__
    if column_1 is None:
        column_1 = f'{table_1[:-1]}_id'
    if column_2 is None:
        column_2 = f'{table_2[:-1]}_id'

    model_attrs = {
        '__tablename__': join_table,
        'id': None,
        column_1: sa.Column(column_1, sa.ForeignKey(f'{table_1}.{fk_1}',
                                                    ondelete='CASCADE'),
                            primary_key=True),
        column_2: sa.Column(column_2, sa.ForeignKey(f'{table_2}.{fk_2}',
                                                    ondelete='CASCADE'),
                            primary_key=True),
    }

    model_attrs.update({
        model_1.__name__.lower(): relationship(model_1, cascade='all',
                                               foreign_keys=[
                                                   model_attrs[column_1]
                                               ]),
        model_2.__name__.lower(): relationship(model_2, cascade='all',
                                               foreign_keys=[
                                                   model_attrs[column_2]
                                               ])
    })
    model = type(model_1.__name__ + model_2.__name__, (base,), model_attrs)

    return model


class SpatiallyIndexed(object):
    """A mixin indicating to the database that an object has sky coordinates.
    Classes that mix this class get a q3c spatial index on ra and dec.

    Columns:
        ra: the icrs right ascension of the object in degrees
        dec: the icrs declination of the object in degrees

    Indexes:
        q3c index on ra, dec

    Properties: skycoord: astropy.coordinates.SkyCoord representation of the
    object's coordinate
    """

    # database-mapped
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)

    @property
    def skycoord(self):
        return SkyCoord(self.ra, self.dec, unit='deg')

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return sa.Index(f'{tn}_q3c_ang2ipix_idx', sa.func.q3c_ang2ipix(
            cls.ra, cls.dec)),


class UnmappedFileError(FileNotFoundError):
    """Error raised when a user attempts to call a method of a `File` that
    requires the file to be mapped to a file on disk, but the file is not
    mapped. """
    pass


class File(object):
    """A python object mappable to a file on spinning disk, with metadata
    mappable to rows in a database.`File`s should be thought of as python
    objects that live in memory that can represent the data and metadata of
    files on disk. If mapped to database records, they can serve as an
    intermediary between database records and files that live on disk.

    `File`s can read and write data and metadata to and from disk and to and
    from the database. However, there are some general rules about this that
    should be heeded to ensure good performance.

    In general, files that live on disk and are represented by `File`s
    contain data that users and subclasses may want to use. It is imperative
    that file metadata, and only file metadata, should be mapped to the
    database by instances of File. Data larger than a few kb from disk-mapped
    files should never be stored in the database. Instead it should reside on
    disk. Putting the file data directly into the database would slow it down
    and make queries take too long.

    Files represented by this class can reside on spinning disk only. This
    class does not make any attempt to represent files that reside on tape.

    The user is responsible for manually associating `File`s  with files on
    disk. This frees `File`s  from being tied to specific blocks of disk on
    specific machines.  The class makes no attempt to enforce that the user
    maps python objects tied to particular database records to the "right"
    files on disk. This is the user's responsibility!

    `property`s of Files should be assumed to represent what is in memory
    only, not necessarily what is on disk. Disk-memory synchronization is up
    to the user and can be achived using the save() function.
    """

    basename = sa.Column(sa.Text, unique=True, index=True)
    __diskmapped_cached_properties__ = ['_path']

    @property
    def local_path(self):
        try:
            return self._path
        except AttributeError:
            errormsg = f'File "{self.basename}" is not mapped to the local ' \
                       f'file system. ' \
                       f'Identify the file corresponding to this object on ' \
                       f'the local file system, ' \
                       f'then call the `map_to_local_file` to identify the ' \
                       f'path. '
            raise UnmappedFileError(errormsg)

    @property
    def ismapped(self):
        return hasattr(self, '_path')

    def map_to_local_file(self, path, quiet=False):
        if not quiet:
            if  hasattr(self, 'basename'):
                print(f'Mapping {self.basename} to {path}')
            else:
                print(f'Mapping {self} to {path}')
        self._path = str(Path(path).absolute())

    def unmap(self):
        if not self.ismapped:
            raise UnmappedFileError(f"Cannot unmap file '{self.basename}', "
                                    f"file is not mapped")
        self.clear()

    def clear(self):
        for attr in self.__diskmapped_cached_properties__:
            if hasattr(self, attr):
                delattr(self, attr)

    def save(self):
        """Update the data and metadata of a mapped file on disk to reflect
        their values in this object."""
        raise NotImplemented

    def load(self):
        """Load the data and metadata of a mapped file on disk into memory
        and set the values of database mapped columns, which can later be
        flushed into the DB."""
        raise NotImplemented


class ZTFFileCopy(models.Base):
    """Record of a *permanent* (i.e., will not be deleted or modified,
    or will only be touched extremely rarely and by someone who can do it in
    concert with the database).

    A Copy is diferent from a File in that a copy cannot be mapped to a file
    on local disk, whereas a File is mappable. A copy is just a record of a
    file that lives in a permanent place somewhere."""

    __tablename__ = 'ztffilecopies'

    __mapper_args__ = {
        'polymorphic_on': 'type',
        'polymorphic_identity': 'copy'
    }

    type = sa.Column(sa.Text)
    product_id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                                     ondelete='CASCADE'),
                           index=True)
    product = relationship('ZTFFile', back_populates='copies',
                           cascade='all')

    def get(self):
        """Pull the Copy to local disk and return the corresponding
        Product (File subclass) that the Copy is mapped to."""
        raise NotImplemented


class HTTPArchiveCopy(ZTFFileCopy):
    """Record of a copy of a ZTFFile that lives on the ZUDS disk
    archive at NERSC (on project) and is accessible via HTTP."""

    __mapper_args__ = {
        'polymorphic_identity': 'http'
    }

    __tablename__ = 'httparchivecopies'

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffilecopies.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    url = sa.Column(sa.Text, index=True, unique=True)
    archive_path = sa.Column(sa.Text)

    def get(self):
        product = self.product
        with open(product.basename, 'wb') as f:
            r = requests.get(self.url)
            r.raise_for_status()
            f.write(r.content)
            product.map_to_local_file(product.basename)
        return product

    def put(self):
        archive.archive(self)

    @classmethod
    def from_product(cls, product, check=True):
        if not isinstance(product, ZTFFile):
            raise ValueError(
                f'Cannot archive object "{product}", must be an instance of'
                f'PipelineFITSProduct.')

        field = product.field
        qid = product.qid
        ccdid = product.ccdid
        fid = product.fid
        band = fid_map[fid]

        path = Path(NERSC_PREFIX) / f'{field:06d}/' \
                                    f'c{ccdid:02d}/' \
                                    f'q{qid}/' \
                                    f'{band}/' \
                                    f'{product.basename}'

        archive_path = f'{path.absolute()}'
        url = f'{path.absolute()}'.replace(NERSC_PREFIX[:-5], URL_PREFIX)

        # check to see if a copy with this URL already exists.
        # if so return it


        if check:
            old = DBSession().query(cls).filter(
                cls.url == url
            ).first()
        else:
            old = None

        if old is None:
            copy = cls()
            copy.archive_path = archive_path
            copy.url = url
            copy.product = product
            return copy

        else:
            return old


class TapeCopy(ZTFFileCopy):
    """Record of a copy of a ZTFFile that lives inside a
    tape archive on HPSS."""

    __tablename__ = 'tapecopies'

    __mapper_args__ = {
        'polymorphic_identity': 'tape'
    }

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffilecopies.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    archive_id = sa.Column(sa.Text, sa.ForeignKey('tapearchives.id',
                                                  ondelete='CASCADE'),
                           index=True)
    archive = relationship('TapeArchive', back_populates='contents')

    # The exact name of the TAR archive member
    member_name = sa.Column(sa.Text)


class TapeArchive(models.Base):
    """Record of a tape archive that contains copies of ZTFFiles."""
    id = sa.Column(sa.Text, primary_key=True)
    contents = relationship('TapeCopy', cascade='all')
    size = sa.Column(psql.BIGINT)  # size of the archive in bytes

    @classmethod
    def from_directories(cls, path):
        raise NotImplemented

    def get(self, extract=False, extract_kws=None):
        # use retrieve.retrieve for now
        raise NotImplemented


class FITSFile(File):
    """A python object that maps a fits file. Instances of classes mixed with
    FITSFile that implement `Base` map to database rows that store fits file
    metadata.

    Assumes the fits file has only one extension. TODO: add support for
    multi-extension fits files.

    Provides methods for loading fits file data from disk to memory,
    and writing it to disk from memory.

    Methods for loading fits file metadata from the database and back are
    handled by sqlalchemy.
    """

    header = sa.Column(psql.JSONB)
    header_comments = sa.Column(psql.JSONB)
    __diskmapped_cached_properties__ = ['_path', '_data']
    _DATA_HDU = 0
    _HEADER_HDU = 0

    @classmethod
    def from_file(cls, f, use_existing_record=True):
        """Read a file into memory from disk, and set the values of
        database-backed variables that store metadata (e.g., header). These
        can later be flushed to the database using SQLalchemy.

        This is a 'get_or_create' method."""
        f = Path(f)

        load_from_db = issubclass(cls, models.Base) and \
                       issubclass(cls, FITSFile) and \
                       use_existing_record

        if load_from_db:
            obj = cls.get_by_basename(f.name)
        else:
            obj = None
        if obj is None:
            obj = cls()
            obj.basename = f.name
        else:
            # this should never have to be called, as things are unmapped in
            # get_by_basename
            if obj.ismapped:
                obj.unmap() #  force things to be reloaded from disk
        obj.map_to_local_file(str(f.absolute()))
        obj.load_header()

        return obj

    def load_header(self):
        """Load a header from disk into memory. Sets the values of
        database-backed variables that store metadata. These can later be
        flushed to the database using SQLalchemy."""
        with fits.open(self.local_path) as hdul:
            hd = dict(hdul[self._HEADER_HDU].header)
            hdc = {card.keyword: card.comment
                   for card in hdul[self._HEADER_HDU].header.cards}
        hd2 = hd.copy()
        for k in hd:
            if not isinstance(hd[k], (int, str, bool, float)):
                del hd2[k]
                del hdc[k]
        self.header = hd2
        self.header_comments = hdc

    def load_data(self):
        """Load data from disk into memory"""
        with fits.open(self.local_path, memmap=False) as hdul:  # throws
            # UnmappedFileError
            data = hdul[self._DATA_HDU].data
        if data.dtype.name == 'uint8':
            data = data.astype(bool)
        self._data = data

    def unload_data(self):
        try:
            del self._data
        except AttributeError:
            raise RuntimeError(f'Object "<{self.__class__.__name__} at '
                               f'{hex(id(self))}>" has no data loaded. '
                               f'Load some data with .load_data() and '
                               f'try again.')

    @property
    def data(self):
        """Data are read directly from a mapped file on disk, and cached in
        memory to avoid unnecessary IO.

        This property can be modified, but in order to save the resulting
        changes to disk the save() method must be called.
        """
        try:
            return self._data
        except AttributeError:
            # load the data into memory
            self.load_data()
        return self._data

    @data.setter
    def data(self, d):
        """Update the data member of this object in memory only. To flush
        this to disk you must call `save`."""
        self._data = d

    @property
    def astropy_header(self):
        """astropy.io.fits.Header representation of the database-mapped
        metadata columns header and header_comments. This attribute may not
        reflect the current header on disk"""
        if self.header is None or self.header_comments is None:
            # haven't seen the file on disk yet
            raise AttributeError(f'This image does not have a header '
                                 f'or headercomments record yet. Please map '
                                 f'this object to the corresponding '
                                 f'ScienceImage file on disk, load the header '
                                 f'with .load_header(), and retry.')
        else:
            header = fits.Header()
            header.update(self.header)
            for key in self.header_comments:
                header.comments[key] = self.header_comments[key]
            return header

    def save(self):
        try:
            f = self.local_path
        except UnmappedFileError:
            f = self.basename
            self.map_to_local_file(f)
        dname = self.data.dtype.name
        if dname == 'bool':
            data = self.data.astype('uint8')
        else:
            data = self.data

        nhdu = max(self._DATA_HDU, self._HEADER_HDU) + 1


        if isinstance(self, FITSImage):
            if nhdu == 1:
                fits.writeto(f, data, self.astropy_header, overwrite=True)
            else:
                hdul = []
                for i in range(nhdu):
                    if i == 0:
                        hdu = fits.PrimaryHDU()
                    else:
                        hdu = fits.ImageHDU()
                    hdul.append(hdu)
                hdul = fits.HDUList(hdul)
                hdul[self._HEADER_HDU].header = self.astropy_header
                hdul[self._DATA_HDU].data = data

                hdul.writeto(f, overwrite=True)
        else:  # it's a catalog
            with fitsio.FITS(f, 'rw', clobber=True) as out:
                for i in range(nhdu):
                    data = None
                    header = None
                    if i == self._DATA_HDU:
                        data = self.data
                    if i == self._HEADER_HDU:
                        header = []
                        for key in self.header:
                            card = {
                                'name': key,
                                'value': self.header[key],
                                'comment': self.header_comments[key]
                            }
                            header.append(card)
                    out.write(data, header=header)


        self.unload_data()

    def load(self):
        self.load_header()
        self.load_data()


class HasPoly(object):
    """Mixin indicating that an object represents an entity with four corners
    on the celestial sphere, connected by great circles.

    The four corners, ra{1..4}, dec{1..4} are database-backed metadata,
    and are thus queryable.

    Provides a hybrid property `poly` (in-memory and in-db), which can be
    used to query against the polygon in the database or to access the
    polygon in memory.
    """

    ra1 = sa.Column(psql.DOUBLE_PRECISION)
    dec1 = sa.Column(psql.DOUBLE_PRECISION)
    ra2 = sa.Column(psql.DOUBLE_PRECISION)
    dec2 = sa.Column(psql.DOUBLE_PRECISION)
    ra3 = sa.Column(psql.DOUBLE_PRECISION)
    dec3 = sa.Column(psql.DOUBLE_PRECISION)
    ra4 = sa.Column(psql.DOUBLE_PRECISION)
    dec4 = sa.Column(psql.DOUBLE_PRECISION)

    @hybrid_property
    def poly(self):
        return array((self.ra1, self.dec1, self.ra2, self.dec2,
                      self.ra3, self.dec3, self.ra4, self.dec4))


def needs_update(obj, key, value):
    # determine if an angle has changed enough in an object to warrant being
    # updated

    if not hasattr(obj, key):
        return True
    else:
        curval = getattr(obj, key)
        try:
            return not np.isclose(curval, value)
        except TypeError:
            # Duck typing - if np.isclose fails then curval is not a number
            # but instead is probably an sqlalchemy Column, thus needs an update
            return True


class HasWCS(FITSFile, HasPoly, SpatiallyIndexed):
    """Mixin indicating that an object represents a fits file with a WCS
    solution."""

    @property
    def wcs(self):
        """Astropy representation of the fits file's WCS solution.
        Lives in memory only."""
        return WCS(self.astropy_header)

    @classmethod
    def from_file(cls, fname, use_existing_record=True):
        """Read a fits file into memory from disk, and set the values of
        database-backed variables that store metadata (e.g., header). These
        can later be flushed to the database using SQLalchemy. """
        self = super(HasWCS, cls).from_file(
            fname, use_existing_record=use_existing_record,
        )
        corners = self.wcs.calc_footprint()
        for i, values in enumerate(corners):
            keys = [f'ra{i+1}', f'dec{i+1}']
            for key, value in zip(keys, values):
                if needs_update(self, key, value):
                    setattr(self, key, value)

        naxis1 = self.header['NAXIS1']
        naxis2 = self.header['NAXIS2']
        ra, dec = self.wcs.all_pix2world(
            [[naxis1 / 2, naxis2 / 2]], 1
        )[0]

        for key, value in zip(['ra', 'dec'], [ra, dec]):
            if needs_update(self, key, value):
                setattr(self, key, value)

        return self

    @property
    def sources_contained(self):
        """Query the database and return all `Sources` contained by the
        polygon of this object"""
        return DBSession().query(models.Source) \
            .filter(func.q3c_poly_query(models.Source.ra,
                                        models.Source.dec,
                                        self.poly))

    @property
    def pixel_scale(self):
        """The pixel scales of the detector in the X and Y image dimensions. """
        units = self.wcs.world_axis_units
        u1 = getattr(u, units[0])
        u2 = getattr(u, units[1])
        scales = proj_plane_pixel_scales(self.wcs)
        ps1 = (scales[0] * u1).to('arcsec').value
        ps2 = (scales[1] * u2).to('arcsec').value
        return np.asarray([ps1, ps2]) * u.arcsec

    def aligned_to(self, other, persist_aligned=False, tmpdir='/tmp',
                   nthreads=1):
        """Return a version of this object that is pixel-by-pixel aligned to
        the WCS solution of another image with a WCS solution."""

        if not isinstance(other, HasWCS):
            raise ValueError(f'WCS Alignment target must be an instance of '
                             f'HasWCS (got "{other.__class__}").')

        new = run_align(self, other,
                        tmpdir=tmpdir,
                        nthreads=nthreads,
                        persist_aligned=persist_aligned)

        if hasattr(self, 'mask_image'):
            newmask = run_align(self.mask_image, other,
                                tmpdir=tmpdir,
                                nthreads=nthreads,
                                persist_aligned=persist_aligned)
            new.mask_image = newmask

        return new


class FITSImage(HasWCS):
    """A `FITSFile` with a data member representing an image. Same as
    FITSFile, but provides the method show() to render the image in
    matplotlib. Also defines some properties that help to optimally render
    the image (cmap, cmap_limits)"""

    def show(self, axis=None, align_to=None):
        if axis is None:
            fig, axis = plt.subplots()

        if align_to is not None:
            image = self.aligned_to(align_to)
        else:
            image = self

        vmin, vmax = image.cmap_limits()

        axis.imshow(image.data,
                    vmin=vmin,
                    vmax=vmax,
                    norm=image.cmap_norm(),
                    cmap=image.cmap(),
                    interpolation='none')

    @property
    def datatype(self):
        dtype = self.data.dtype.name
        if 'float' in dtype:
            return 'float'
        else:
            return 'int'

    def cmap_limits(self):
        if self.datatype == 'float':
            interval = ZScaleInterval()
            return interval.get_limits(self.data)
        else:  # integer
            return (None, None)

    def cmap(self):
        if self.datatype == 'float':
            return 'gray'
        else:
            ncolors = len(np.unique(self.data))
            return discrete_cmap(ncolors)

    def cmap_norm(self):
        if self.datatype == 'float':
            return None
        else:
            boundaries = np.unique(self.data)
            ncolors = len(boundaries)
            return colors.BoundaryNorm(boundaries, ncolors)


class ZTFFile(models.Base, File):
    """A database-mapped, disk-mappable memory-representation of a file that
    is associated with a ZTF sky partition. This class is abstract and not
    designed to be instantiated, but it is also not a mixin. Think of it as a
    base class for the polymorphic hierarchy of products in SQLalchemy.

    To create an disk-mappable representation of a fits file that stores data in
    memory and is not mapped to rows in the database, instantiate FITSFile
    directly.
    """

    # this is the discriminator that is used to keep track of different types
    #  of fits files produced by the pipeline for the rest of the hierarchy
    type = sa.Column(sa.Text)

    # all pipeline fits products must implement these four key pieces of
    # metadata. These are all assumed to be not None in valid instances of
    # ZTFFile.

    field = sa.Column(sa.Integer)
    qid = sa.Column(sa.Integer)
    fid = sa.Column(sa.Integer)
    ccdid = sa.Column(sa.Integer)

    copies = relationship('ZTFFileCopy', cascade='all')

    # An index on the four indentifying
    idx = sa.Index('fitsproduct_field_ccdid_qid_fid', field, ccdid, qid, fid)

    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'fitsproduct'

    }

    def find_in_dir(self, directory):
        target = os.path.join(directory, self.basename)
        if os.path.exists(target):
            self.map_to_local_file(target)
        else:
            raise FileNotFoundError(
                f'Cannot map "{self.basename}" to "{target}", '
                f'file does not exist.'
            )

    def find_in_dir_of(self, ztffile):
        dirname = os.path.dirname(ztffile.local_path)
        self.find_in_dir(dirname)

    @classmethod
    def get_by_basename(cls, basename):
        obj = DBSession().query(cls).filter(cls.basename == basename).first()
        if obj is not None:
            obj.clear()  # get a fresh copy

        if hasattr(obj, 'mask_image'):
            if obj.mask_image is not None:
                obj.mask_image.clear()

        if hasattr(obj, 'catalog'):
            if obj.catalog is not None:
                obj.catalog.clear()

        return obj



class PipelineRegionFile(ZTFFile):
    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    __mapper_args__ = {
        'polymorphic_identity': 'regionfile',
        'inherit_condition': id == ZTFFile.id
    }

    catalog_id = sa.Column(sa.Integer, sa.ForeignKey(
        'pipelinefitscatalogs.id', ondelete='CASCADE'
    ), index=True)
    catalog = relationship('PipelineFITSCatalog', cascade='all',
                           foreign_keys=[catalog_id],
                           back_populates='regionfile')

    @classmethod
    def from_catalog(cls, catalog):
        basename = catalog.basename.replace('.cat', '.reg')
        reg = cls.get_by_basename(basename)
        if reg is None:
            reg = cls()
            reg.basename = basename

        catdir = os.path.dirname(catalog.local_path)
        reg.map_to_local_file(os.path.join(catdir, reg.basename))

        reg.field = catalog.field
        reg.ccdid = catalog.ccdid
        reg.qid = catalog.qid
        reg.fid = catalog.fid

        filtered = 'GOODCUT' in catalog.data.dtype.names

        reg.catalog = catalog
        with open(reg.local_path, 'w') as f:
            f.write('global color=green dashlist=8 3 width=1 font="helvetica '
                    '10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 '
                    'delete=1 include=1 source=1\n')
            f.write('icrs\n')
            rad = 13 * catalog.image.pixel_scale.to(
                'arcsec'
            ).value.mean() * 0.00027777
            for row in catalog.data:
                if not filtered:
                    color = 'blue'
                else:
                    color = 'green' if row['GOODCUT'] else 'red'

                f.write(f'circle({row["X_WORLD"]},'
                        f'{row["Y_WORLD"]},{rad}) # width=2 '
                        f'color={color}\n')

        return reg


models.Thumbnail.image_id = sa.Column(
    sa.Integer,
    sa.ForeignKey(
        'calibratableimages.id',
        ondelete='CASCADE'
    ),
    index=True,
    nullable=True
)


models.Thumbnail.source_id = sa.Column(
    sa.Text,
    sa.ForeignKey(
        'sources.id',
        ondelete='CASCADE'
    ),
    index=True,
    nullable=True
)

models.Thumbnail.detection_id = sa.Column(
    sa.Integer,
    sa.ForeignKey(
        'detections.id',
        ondelete='CASCADE'
    ),
    index=True,
    nullable=True
)

import io
import gzip


def from_detection(cls, detection, image):

    if isinstance(image, models.Base):
        linkimage = image
    else:
        linkimage = image.parent_image

    stamp = cls(image=linkimage, detection=detection)

    if isinstance(linkimage, Subtraction):
        stamp.type = 'sub'
    elif isinstance(linkimage, ReferenceImage):
        stamp.type = 'ref'
    else:
        stamp.type = 'new'

    cutout = publish.make_stamp(
        None, detection.ra, detection.dec, None,
        None, image.data, image.wcs, save=False,
        size=publish.CUTOUT_SIZE
    )

    # convert the cutout data to bytes
    # and store that in the database

    fitsbuf = io.BytesIO()
    gzbuf = io.BytesIO()

    fits.writeto(fitsbuf, cutout.data, header=cutout.wcs.to_header())
    with gzip.open(gzbuf, 'wb') as fz:
        fz.write(fitsbuf.getvalue())

    stamp.bytes = gzbuf.getvalue()
    stamp.detection = detection

    return stamp

def persist(self):

    if os.getenv('NERSC_HOST') != 'cori':
        raise RuntimeError('Must be on cori to persist stamps.')

    data = self.array
    self.source = self.detection.source
    vmin, vmax = ZScaleInterval().get_limits(data)
    img = self.image
    base = f'stamp.{self.id}.jpg'
    relpath = f'stamps/{img.field:06d}/c{img.ccdid:02d}/' \
              f'q{img.qid}/{fid_map[img.fid]}/{base}'
    name = Path(URL_PREFIX) / relpath
    self.public_url = f'{name}'
    self.file_uri = f'{Path(STAMP_PREFIX) / relpath}'
    archive._mkdir_recursive(Path(self.file_uri).parent)

    plt.imsave(self.file_uri, data, vmin=vmin, vmax=vmax,
               cmap='gray')

    os.chmod(self.file_uri, 0o774)


models.Thumbnail.from_detection = classmethod(from_detection)
models.Thumbnail.bytes = deferred(sa.Column(psql.BYTEA))
models.Thumbnail.persist = persist


class PipelineFITSCatalog(ZTFFile, FITSFile):
    """Python object that maps a catalog stored on a fits file on disk."""

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'catalog',
        'inherit_condition': id == ZTFFile.id
    }

    image_id = sa.Column(sa.Integer,
                         sa.ForeignKey('calibratableimages.id',
                                       ondelete='CASCADE'),
                         index=True)
    image = relationship('CalibratableImage', cascade='all',
                         foreign_keys=[image_id])

    regionfile = relationship('PipelineRegionFile', cascade='all',
                              uselist=False,
                              primaryjoin=PipelineRegionFile.catalog_id == id)

    # since this object maps a fits binary table, the data lives in the first
    #  extension, not in the primary hdu
    _DATA_HDU = 2
    _HEADER_HDU = 2

    @classmethod
    def from_image(cls, image, tmpdir='/tmp', kill_flagged=True):
        if not isinstance(image, CalibratableImage):
            raise ValueError('Image is not an instance of '
                             'CalibratableImage.')

        image._call_source_extractor(tmpdir=tmpdir)
        cat = image.catalog

        for prop in GROUP_PROPERTIES:
            setattr(cat, prop, getattr(image, prop))

        df = pd.DataFrame(cat.data)
        rec = df.to_records(index=False)
        cat.data = rec
        cat.basename = image.basename.replace('.fits', '.cat')
        cat.image_id = image.id
        cat.image = image
        image.catalog = cat

        if kill_flagged:
            image.catalog.kill_flagged()

        return cat

    def kill_flagged(self):
        # overwrite the catalog, killing any detections with bad IMAFLAGS_ISO
        self.load()
        oinds = []
        for i, row in enumerate(self.data):
            if row['IMAFLAGS_ISO'] & BAD_SUM == 0 and row['FLAGS_WEIGHT'] == 0:
                oinds.append(i)
        out = self.data[oinds]
        self.data = out
        self.save()
        self.load()


class MaskImageBase(FITSImage):

    __diskmapped_cached_properties__ = FITSImage.__diskmapped_cached_properties__ + [
        '_boolean']


    def refresh_bit_mask_entries_in_header(self):
        """Update the database record and the disk record with a constant
        bitmap information stored in in memory. """
        self.header.update(MASK_BITS)
        self.header_comments.update(MASK_COMMENTS)
        self.save()

    def update_from_weight_map(self, weight_image):
        """Update what's in memory based on what's on disk (produced by
        SWarp)."""
        mskarr = self.data
        ftsarr = weight_image.data
        mskarr[ftsarr == 0] += 2 ** 16
        self.data = mskarr
        self.refresh_bit_mask_entries_in_header()

    @classmethod
    def from_file(cls, f, use_existing_record=True):
        return super().from_file(
            f, use_existing_record=use_existing_record,
        )

    @property
    def boolean(self):
        """A boolean array that is True when a masked pixel is 'bad', i.e.,
        not usable for science, and False when a pixel is not masked."""
        try:
            return self._boolean
        except AttributeError:

            # From ZSDS document:

            # For example, to find all science - image pixels which are
            # uncontaminated, but which may contain clean extracted - source
            # signal(bits 1 or 11), one would “logically AND” the
            # corresponding mask - image with the template value 6141 (= 2^0
            # + 2^2 +2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9 + 2^10 + 21^2)
            # #and retain pixels where this operation yields zero. To then
            # find which of these pixels contain source signal, one would
            # “AND” the resulting image with 2050 (= 21 + 211) and retain
            # pixels where this operation is non-zero.

            # DG: also have to add 2^16 (my custom bit) and 2^17 (hotpants
            # masked, another of my custom bits)
            # So 6141 -> 71677 --> 202749

            maskpix = (self.data & BAD_SUM) > 0
            _boolean = FITSImage()
            _boolean.data = maskpix
            _boolean.header = self.header
            _boolean.header_comments = self.header_comments
            _boolean.basename = self.basename.replace('.fits', '.bpm.fits')
            self._boolean = _boolean
        return self._boolean


class MaskImage(ZTFFile, MaskImageBase):

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'mask',
        'inherit_condition': id == ZTFFile.id
    }

    parent_image_id = sa.Column(sa.Integer,
                                sa.ForeignKey('calibratableimages.id',
                                              ondelete='CASCADE'))
    parent_image = relationship('CalibratableImage', cascade='all',
                                back_populates='mask_image',
                                foreign_keys=[parent_image_id])

    idx = Index('maskimages_parent_image_id_idx', parent_image_id)

    pass


class CalibratableImageBase(FITSImage):
    __diskmapped_cached_properties__ = ['_path', '_data', '_weightimg',
                                        '_bkgimg', '_filter_kernel', '_rmsimg',
                                        '_threshimg', '_segmimg',
                                        '_sourcelist', '_bkgsubimg']


    def cmap_limits(self):
        interval = ZScaleInterval()
        return interval.get_limits(self.data[~self.mask_image.boolean.data])

    def _call_source_extractor(self, checkimage_type=None, tmpdir='/tmp',
                               use_weightmap=True):

        rs = sextractor.run_sextractor
        success = False
        for _ in range(3):
            try:
                results = rs(
                    self, checkimage_type=checkimage_type, tmpdir=tmpdir,
                    use_weightmap=use_weightmap
                )
            except subprocess.CalledProcessError as e:
                print(f'Caught CalledProcessError {e}, retrying... {_+1} / 3')
                continue
            else:
                success = True
                break

        if not success:
            raise ValueError(f'Unable to run SExtractor on {self}...')

        for result in results:
            if result.basename.endswith('.cat'):
                self.catalog = result
            elif result.basename.endswith('.rms.fits'):
                self._rmsimg = result
            elif result.basename.endswith('.bkg.fits'):
                self._bkgimg = result
            elif result.basename.endswith('.bkgsub.fits'):
                self._bkgsubimg = result
            elif result.basename.endswith('.segm.fits'):
                self._segmimg = result

    @property
    def weight_image(self):
        """Image representing the inverse variance map of this calibratable
        image."""

        try:
            return self._weightimg
        except AttributeError:
            # need to calculate the weight map.

            ind = self.mask_image.boolean.data
            wgt = np.empty_like(ind, dtype='<f4')
            wgt[~ind] = 1 / self.rms_image.data[~ind] ** 2
            wgt[ind] = 0.

            try:
                saturval = self.header['SATURATE']
            except KeyError:
                pass
            else:
                saturind = self.data >= 0.9 * saturval
                wgt[saturind] = 0.

            self._weightimg = FITSImage()
            self._weightimg.basename = self.basename.replace('.fits',
                                                             '.weight.fits')
            self._weightimg.data = wgt
            self._weightimg.header = self.header
            self._weightimg.header_comments = self.header_comments
            if self.ismapped:
                dirname = os.path.dirname(self.local_path)
                bn = self._weightimg.basename
                join = os.path.join(dirname, bn)
                self._weightimg.map_to_local_file(join)
                self._weightimg.save()  #  to guarantee mapped file exists
        return self._weightimg

    @property
    def rms_image(self):
        try:
            return self._rmsimg
        except AttributeError:
            if hasattr(self, '_weightimg'):
                ind = self.mask_image.boolean.data
                rms = np.empty_like(ind, dtype='<f4')
                rms[~ind] = 1 / np.sqrt(self.weight_image.data[~ind])
                rms[ind] = BIG_RMS

                try:
                    saturval = self.header['SATURATE']
                except KeyError:
                    pass
                else:
                    saturind = self.data >= 0.9 * saturval
                    rms[saturind] = BIG_RMS

                _rmsimg = FITSImage()
                _rmsimg.basename = self.basename.replace('.fits', '.rms.fits')
                _rmsimg.data = rms
                _rmsimg.header = self.header
                _rmsimg.header_comments = self.header_comments
                if self.ismapped:
                    dirname = os.path.dirname(self.local_path)
                    bn = _rmsimg.basename
                    join = os.path.join(dirname, bn)
                    _rmsimg.map_to_local_file(join)
                    _rmsimg.save()
                self._rmsimg = _rmsimg
                return self._rmsimg
            else:
                self._call_source_extractor(checkimage_type=['rms'],
                                            use_weightmap=False)
        return self._rmsimg

    @property
    def background_image(self):
        try:
            return self._bkgimg
        except AttributeError:
            self._call_source_extractor(checkimage_type=['bkg'])
        return self._bkgimg

    @property
    def background_subtracted_image(self):
        try:
            return self._bkgsubimg
        except AttributeError:
            self._call_source_extractor(checkimage_type=['bkgsub'])
        return self._bkgsubimg

    @property
    def segm_image(self):
        try:
            return self._segmimg
        except AttributeError:
            # segm is set here
            self._call_source_extractor(checkimage_type=['segm'])
        return self._segmimg


    @classmethod
    def from_file(cls, fname, use_existing_record=True, load_others=True):
        obj = super().from_file(
            fname, use_existing_record=use_existing_record,
        )
        dir = Path(fname).parent

        if load_others:
            weightpath = dir / obj.basename.replace('.fits', '.weight.fits')
            rmspath = dir / obj.basename.replace('.fits', '.rms.fits')
            bkgpath = dir / obj.basename.replace('.fits', '.bkg.fits')
            threshpath = dir / obj.basename.replace('.fits', '.thresh.fits')
            bkgsubpath = dir / obj.basename.replace('.fits', '.bkgsub.fits')
            segmpath = dir / obj.basename.replace('.fits', '.segm.fits')

            paths = [weightpath, rmspath, bkgpath, threshpath,
                     bkgsubpath, segmpath]

            types = ['_weightimg', '_rmsimg', '_bkgimg', '_threshimg',
                     '_bkgsubimg', '_segmimg']

            for path, t in zip(paths, types):
                if path.exists():
                    print(f'Reading {obj.basename}.{t} from {path}')
                    setattr(obj, t, FITSImage.from_file(f'{path}'))

        return obj


class CalibratableImage(CalibratableImageBase, ZTFFile):

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    __mapper_args__ = {'polymorphic_identity': 'calibratableimage',
                       'inherit_condition': id == ZTFFile.id}

    detections = relationship('Detection', cascade='all')
    objects = relationship('ObjectWithFlux', cascade='all')

    mask_image = relationship('MaskImage',
                              uselist=False,
                              primaryjoin=MaskImage.parent_image_id == id)

    catalog = relationship('PipelineFITSCatalog', uselist=False,
                           primaryjoin=PipelineFITSCatalog.image_id == id)

    thumbnails = relationship('Thumbnail',
                              primaryjoin=models.Thumbnail.image_id == id)


    def basic_map(self, quiet=False):
        datadir = Path(get_secret('base_data_directory'))
        self.map_to_local_file(datadir / self.relname, quiet=quiet)
        self.mask_image.map_to_local_file(datadir / self.mask_image.relname,
                                          quiet=quiet)
        weightpath = datadir / self.relname.replace('.fits', '.weight.fits')
        if weightpath.exists():
            self._weightimg = FITSImage()
            self.weight_image.map_to_local_file(weightpath, quiet=quiet)

        else:
            rmspath = datadir / self.relname.replace('.fits', '.rms.fits')
            if rmspath.exists():
                self.rms_image.map_to_local_file(rmspath, quiet=quiet)
            else:
                raise FileNotFoundError(f'Neither "{weightpath}" nor '
                                        f'"{rmspath}" exists for '
                                        f'"{self.basename}".')


    @classmethod
    def from_file(cls, fname, use_existing_record=True, load_others=True):
        obj = super().from_file(
            fname, use_existing_record=use_existing_record,
            load_others=load_others
        )
        dir = Path(fname).parent

        if load_others:

            if obj.mask_image is not None:
                mskpath = dir / obj.mask_image.basename
                if mskpath.exists():
                    print(f'Reading {obj.basename}.mask_image from {mskpath}')
                    obj.mask_image = MaskImage.from_file(mskpath)

            if obj.catalog is not None:
                catpath = dir / obj.catalog.basename
                if catpath.exists():
                    print(f'Reading {obj.basename}.catalog from {catpath}')
                    obj.catalog = PipelineFITSCatalog.from_file(catpath)

        return obj


class CalibratedImage(CalibratableImage):
    """An image on which photometry can be performed."""

    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'calibratedimage',
                       'inherit_condition': id == CalibratableImage.id}

    forced_photometry = relationship('ForcedPhotometry', cascade='all')

    def force_photometry(self, sources, assume_background_subtracted=False,
                         use_cutout=False, direct_load=None):
        """Force aperture photometry at the locations of `sources`.
        Assumes that calibration has already been done.

        """

        # ensure sources is at least 1d
        sources = np.atleast_1d(sources)

        ra = [source.ra for source in sources]
        dec = [source.dec for source in sources]

        result = aperture_photometry(
            self, ra, dec, apply_calibration=True,
            assume_background_subtracted=assume_background_subtracted,
            use_cutout=use_cutout, direct_load=direct_load
        )

        photometry = []
        for row, source, r, d in zip(result, sources, ra, dec):
            phot = ForcedPhotometry(flux=row['flux'],
                                    fluxerr=row['fluxerr'],
                                    flags=int(row['flags']),
                                    image=self,
                                    ra=r,
                                    dec=d,
                                    source=source,
                                    zp=row['zp'],
                                    obsjd=row['obsjd'],
                                    filtercode=row['filtercode'])
            photometry.append(phot)

        return photometry

    @declared_attr
    def __table_args__(cls):
        # override spatial index - only need it on one table (calibratable images)
        return tuple()

    @hybrid_property
    def seeing(self):
        """FWHM of seeing in pixels."""
        return self.header['SEEING']

    @hybrid_property
    def magzp(self):
        return self.header['MAGZP']

    @hybrid_property
    def apcor(self):
        return self.header[APER_KEY]

    @property
    def mjd(self):
        pass

    def find_in_dir(self, directory):
        super().find_in_dir(directory)
        try:
            self.mask_image.find_in_dir(directory)
        except FileNotFoundError:
            pass

    @property
    def unphotometered_sources(self):

        jcond2 = sa.and_(
            ForcedPhotometry.image_id == self.id,
            ForcedPhotometry.source_id == models.Source.id
        )

        query = DBSession().query(
            models.Source
        ).outerjoin(
            ForcedPhotometry, jcond2
        ).filter(
            ForcedPhotometry.id == None
        ).filter(
            sa.func.q3c_poly_query(
                models.Source.ra,
                models.Source.dec,
                self.poly
            )
        )#.with_for_update(of=models.Source)

        return query.all()



class ScienceImage(CalibratedImage):
    """IPAC record of a science image from their pipeline. Contains some
    metadata that IPAC makes available through its irsa metadata query
    service.  This class is primarily intended to enable the reflection of
    IPAC's idea of its science images and which science images exist so that
    IPAC's world can be compared against the results of this pipeline.

    This class does not map to any file on disk, in the sense that it is not
    designed to reflect the data or metadata of any local file to memory,
    but it can be used to download files from the IRSA archive to disk (
    again, it does not make any attempt to represent the contents of these
    files, or synchronize their contents on disk to their representation in
    memory).

    This class represents immutable metadata only.
    """

    # __tablename__ = 'ipacrecords'

    # we dont want science image records to be deleted in a cascade.
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id',
                                             ondelete='RESTRICT'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'sci',
                       'inherit_condition': id == CalibratedImage.id}

    @classmethod
    def from_file(cls, f, use_existing_record=True, load_others=True):
        obj = super().from_file(
            f, use_existing_record=use_existing_record,
            load_others=load_others
        )
        obj.field = obj.header['FIELDID']
        obj.ccdid = obj.header['CCDID']
        obj.qid = obj.header['QID']
        obj.fid = obj.header['FILTERID']
        return obj

    filtercode = sa.Column(sa.CHAR(2))
    obsjd = sa.Column(psql.DOUBLE_PRECISION)
    infobits = sa.Column(sa.Integer)
    pid = sa.Column(psql.BIGINT)
    nid = sa.Column(sa.Integer)
    expid = sa.Column(sa.Integer)
    itid = sa.Column(sa.Integer)
    obsdate = sa.Column(sa.DateTime)
    seeing = sa.Column(sa.Float)
    airmass = sa.Column(sa.Float)
    moonillf = sa.Column(sa.Float)
    moonesb = sa.Column(sa.Float)
    maglimit = sa.Column(sa.Float)
    crpix1 = sa.Column(sa.Float)
    crpix2 = sa.Column(sa.Float)
    crval1 = sa.Column(sa.Float)
    crval2 = sa.Column(sa.Float)
    cd11 = sa.Column(sa.Float)
    cd12 = sa.Column(sa.Float)
    cd21 = sa.Column(sa.Float)
    cd22 = sa.Column(sa.Float)
    ipac_gid = sa.Column(sa.Integer)
    imgtypecode = sa.Column(sa.CHAR(1))
    exptime = sa.Column(sa.Float)
    filefracday = sa.Column(psql.BIGINT)

    @hybrid_property
    def obsmjd(self):
        return self.obsjd - 2400000.5

    @property
    def mjd(self):
        return self.obsmjd

    @hybrid_property
    def filter(self):
        return 'ztf' + self.filtercode[-1]

    def ipac_path(self, suffix):
        """The url of a particular file corresponding to this metadata
        record, with suffix `suffix`, in the IPAC archive. """
        sffd = str(self.filefracday)
        return f'https://irsa.ipac.caltech.edu/ibe/data/ztf/' \
               f'products/sci/{sffd[:4]}/{sffd[4:8]}/{sffd[8:]}/' \
               f'ztf_{sffd}_{self.field:06d}_' \
               f'{self.filtercode}_c{self.ccdid:02d}_' \
               f'{self.imgtypecode}_q{self.qid}_{suffix}'


# Coadds #######################################################################

def _coadd_from_images(cls, images, outfile_name, nthreads=1, data_product=False,
                       tmpdir='/tmp', copy_inputs=False, swarp_kws=None,
                       calculate_seeing=True, addbkg=True):
    """Make a coadd from a bunch of input images"""

    images = np.atleast_1d(images)
    mskoutname = outfile_name.replace('.fits', '.mask.fits')

    basename = os.path.basename(outfile_name)

    # see if a file with this name already exists in the DB
    cond = cls.basename == basename
    predecessor = DBSession().query(cls).filter(cond).first()

    if predecessor is not None:
        warnings.warn(f'WARNING: A "{cls}" object with the basename '
                      f'"{basename}" already exists. The record will be '
                      f'updated...')

    properties = GROUP_PROPERTIES

    # make sure all images have the same field, filter, ccdid, qid:
    ensure_images_have_the_same_properties(images, properties)

    # call swarp
    coadd = run_coadd(cls, images, outfile_name, mskoutname,
                      addbkg=addbkg, nthreads=nthreads,
                      tmpdir=tmpdir, copy_inputs=copy_inputs,
                      swarp_kws=swarp_kws)
    coaddmask = coadd.mask_image

    coadd.header['FIELD'] = coadd.field = images[0].field
    coadd.header['CCDID'] = coadd.ccdid = images[0].ccdid
    coadd.header['QID'] = coadd.qid = images[0].qid
    coadd.header['FID'] = coadd.fid = images[0].fid


    if calculate_seeing:
        estimate_seeing(coadd)
    coadd.save()

    if data_product:
        coadd_copy = HTTPArchiveCopy.from_product(coadd)
        coaddmask_copy = HTTPArchiveCopy.from_product(coaddmask)
        archive.archive(coadd_copy)
        archive.archive(coaddmask_copy)

    return coadd

class Coadd(CalibratableImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'coadd',
        'inherit_condition': id == CalibratableImage.id
    }

    input_images = relationship('CalibratableImage',
                                secondary='coadd_images',
                                cascade='all')


    @property
    def mjd(self):
        return DBSession().query(
            sa.func.percentile_cont(
                0.5
            ).within_group(
                (ScienceImage.obsjd - 2400000.5).asc()
            )
        ).join(
            CoaddImage, CoaddImage.calibratableimage_id == ScienceImage.id
        ).join(
            type(self), type(self).id == CoaddImage.coadd_id
        ).filter(
            type(self).id == self.id
        ).first()[0]

    @property
    def min_mjd(self):
        images = self.input_images
        return min([image.mjd for image in images])

    @property
    def max_mjd(self):
        images = self.input_images
        return max([image.mjd for image in images])


    @declared_attr
    def __table_args__(cls):
        return tuple()

    from_images = classmethod(_coadd_from_images)


CoaddImage = join_model('coadd_images', Coadd, CalibratableImage)


class ReferenceImage(Coadd):
    id = sa.Column(sa.Integer, sa.ForeignKey('coadds.id', ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'ref',
                       'inherit_condition': id == Coadd.id}


    version = sa.Column(sa.Text)

    single_epoch_subtractions = relationship('SingleEpochSubtraction',
                                             cascade='all')
    multi_epoch_subtractions = relationship('MultiEpochSubtraction',
                                            cascade='all')


class ScienceCoadd(Coadd):
    id = sa.Column(sa.Integer, sa.ForeignKey('coadds.id', ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'scicoadd',
                       'inherit_condition': id == Coadd.id}
    subtraction = relationship('MultiEpochSubtraction', uselist=False,
                               cascade='all')


    binleft = sa.Column(sa.DateTime(timezone=False), nullable=False)
    binright = sa.Column(sa.DateTime(timezone=False), nullable=False)

    @hybrid_property
    def winsize(self):
        return self.binright - self.binleft



# Subtractions ################################################################

class Subtraction(HasWCS):

    @declared_attr
    def reference_image_id(self):
        return sa.Column(sa.Integer, sa.ForeignKey('referenceimages.id',
                                                   ondelete='CASCADE'),
                         index=True)

    @declared_attr
    def reference_image(self):
        return relationship('ReferenceImage', cascade='all',
                            foreign_keys=[self.reference_image_id])

    @property
    def mjd(self):
        return self.target_image.mjd

    @classmethod
    def from_images(cls, sci, ref, data_product=False, tmpdir='/tmp',
                    **kwargs):

        refined = kwargs.get('refined', True)

        directory = Path(tmpdir) / uuid.uuid4().hex
        directory.mkdir(exist_ok=True, parents=True)

        # we are gonna copy the inputs into the directory, then reload them
        # off of disk to keep things transactionally isolated
        shutil.copy(sci.local_path, directory)
        shutil.copy(sci.mask_image.local_path, directory)

        if hasattr(sci, '_rmsimg'):
            shutil.copy(sci.rms_image.local_path, directory)
        elif hasattr(sci, '_weightimg'):
            shutil.copy(sci.weight_image.local_path, directory)
        else:
            raise ValueError('Science image must have a weight map or '
                             'rms map defined prior to subtraction.')

        shutil.copy(ref.local_path, directory)
        shutil.copy(ref.mask_image.local_path, directory)
        shutil.copy(ref.weight_image.local_path, directory)

        sciname = os.path.join(directory, sci.basename)
        scimaskn = os.path.join(directory, sci.mask_image.basename)
        refname = os.path.join(directory, ref.basename)
        refmaskn = os.path.join(directory, ref.mask_image.basename)

        transact_sci = CalibratableImageBase.from_file(sciname)
        transact_scimask = MaskImageBase.from_file(scimaskn)
        transact_ref = CalibratableImageBase.from_file(refname)
        transact_refmask = MaskImageBase.from_file(refmaskn)

        transact_sci.mask_image = transact_scimask
        transact_ref.mask_image = transact_refmask

        # output goes into the temporary directory to keep things transactional
        outname = sub_name(transact_sci.local_path, transact_ref.local_path)
        outmask = outname.replace('.fits', '.mask.fits')

        # create the remapped ref, and remapped ref mask. the former will be
        # pixel-by-pixel subtracted from the science image. both will be written
        # to this subtraction's working directory (i.e., `directory`)

        remapped_ref = transact_ref.aligned_to(transact_sci, tmpdir=tmpdir)
        remapped_refmask = remapped_ref.mask_image

        remapped_refname = str(directory / remapped_ref.basename)
        remapped_refmaskname = remapped_refname.replace('.fits', '.mask.fits')

        # flush to the disk
        remapped_ref.map_to_local_file(remapped_refname)
        remapped_refmask.map_to_local_file(remapped_refmaskname)
        remapped_ref.save()
        remapped_refmask.save()
        remapped_ref.parent_image = transact_ref

        # create the mask

        # we truly want this to be a totally new copy of mask, with no
        # uncommitted changes from previous runs
        submask = MaskImageBase()

        submask.basename = os.path.basename(outmask)
        submask.field = sci.field
        submask.ccdid = sci.ccdid
        submask.qid = sci.qid
        submask.fid = sci.fid

        submask.map_to_local_file(outmask)
        badpix = remapped_refmask.data | sci.mask_image.data
        submask.data = badpix
        submask.header = sci.mask_image.header
        submask.header_comments = sci.mask_image.header_comments
        submask.save()

        submask.boolean.map_to_local_file(directory / submask.boolean.basename)
        submask.boolean.save()

        command = prepare_hotpants(transact_sci, remapped_ref, outname,
                                   submask.boolean, directory, tmpdir=tmpdir,
                                   refined=refined)

        final_dir = os.path.dirname(sci.local_path)
        final_out = os.path.join(final_dir, os.path.basename(outname))
        product_map = {
            outname: final_out,
            outname.replace('.fits', '.rms.fits'): final_out.replace('.fits',
                                                                     '.rms.fits'),
            outname.replace('.fits', '.mask.fits'): final_out.replace(
                '.fits', '.mask.fits')
        }

        # run HOTPANTS
        subprocess.check_call(command.split())

        # now modify the sub mask to include the stuff that's masked out from
        # hotpants

        with fits.open(outname) as hdul:
            sd = hdul[0].data

        hotbad = np.zeros_like(submask.data, dtype=int)
        hotbad[sd == 1e-30] = 2**17

        # flip the bits
        submask.data |= hotbad
        submask.header['BIT17'] = 17
        submask.header_comments['BIT17'] = 'MASKED BY HOTPANTS (1e-30) / DG'
        submask.save()

        # now copy the output files to the target directory, ending the
        # transaction

        for f in product_map:
            shutil.copy(f, product_map[f])

        # now read the final output products into database mapped records
        sub = cls.from_file(final_out, load_others=False)
        finalsubmask = MaskImage.from_file(final_out.replace('.fits',
                                                             '.mask.fits'))

        sub._rmsimg = FITSImage.from_file(final_out.replace('.fits',
                                                            '.rms.fits'))

        sub.header['FIELD'] = sub.field = sci.field
        sub.header['CCDID'] = sub.ccdid = sci.ccdid
        sub.header['QID'] = sub.qid = sci.qid
        sub.header['FID'] = sub.fid = sci.fid

        finalsubmask.header['FIELD'] = finalsubmask.field = sci.field
        finalsubmask.header['CCDID'] = finalsubmask.ccdid = sci.ccdid
        finalsubmask.header['QID'] = finalsubmask.qid = sci.qid
        finalsubmask.header['FID'] = finalsubmask.fid = sci.fid

        sub.mask_image = finalsubmask
        sub.reference_image = ref
        sub.target_image = sci

        sub.header['SEEING'] = sci.header['SEEING']
        sub.header_comments['SEEING'] = sci.header_comments['SEEING']


        if isinstance(sub, CalibratedImage):
            sub.header['MAGZP'] = sci.header['MAGZP']
            sub.header[APER_KEY] = sci.header[APER_KEY]
            sub.header_comments['MAGZP'] = sci.header_comments['MAGZP']
            sub.header_comments[APER_KEY] = sci.header_comments[APER_KEY]

        sub.save()
        sub.mask_image.save()

        if data_product:
            archive.archive(sub)
            archive.archive(sub.mask_image)

        shutil.rmtree(directory)

        return sub


class SingleEpochSubtraction(Subtraction, CalibratedImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'sesub',
                       'inherit_condition': id == CalibratedImage.id}

    target_image_id = sa.Column(sa.Integer, sa.ForeignKey('scienceimages.id',
                                                          ondelete='CASCADE'),
                                index=True)
    target_image = relationship('ScienceImage', cascade='all',
                                foreign_keys=[target_image_id])


def overlapping_subtractions(sci, ref):
    subq = DBSession().query(
        SingleEpochSubtraction
    ).join(
        ScienceImage.__table__, ScienceImage.id == SingleEpochSubtraction.target_image_id
    ).join(
        CoaddImage, CoaddImage.calibratableimage_id == ScienceImage.id
    ).filter(
        CoaddImage.coadd_id == sci.id,
        SingleEpochSubtraction.reference_image_id == ref.id
    )

    return subq.all()



class MultiEpochSubtraction(Subtraction, CalibratableImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'mesub',
                       'inherit_condition': id == CalibratableImage.id}
    target_image_id = sa.Column(sa.Integer, sa.ForeignKey('sciencecoadds.id',
                                                          ondelete='CASCADE'),
                                index=True)
    target_image = relationship('ScienceCoadd', cascade='all',
                                foreign_keys=[target_image_id])

    input_images = relationship('SingleEpochSubtraction',
                                secondary='stackedsubtraction_frames',
                                cascade='all')


    @declared_attr
    def __table_args__(cls):
        return tuple()


    @classmethod
    def from_images(cls, sci, ref, data_product=False, tmpdir='/tmp',
                    **kwargs):

        nthreads = kwargs.get('nthreads', 1)
        force_map_subs = kwargs.get('force_map_subs', True)

        if not isinstance(sci, ScienceCoadd):
            raise TypeError(f'Input science image "{sci.basename}" must be '
                            f'an instance of ScienceCoadd, got {type(sci)}.')

        images = overlapping_subtractions(sci, ref)

        if force_map_subs:
            for image in images:
                image.basic_map()

        if len(images) != len(sci.input_images):
            raise ValueError('Number of single-epoch subtractions != number'
                             f' of stack inputs. Stack inputs: '
                             f'{[i.basename for i in sci.input_images]}, '
                             f'Single-epoch subtractions: '
                             f'{[i.basename for i in images]}')

        outfile_name = sub_name(sci.local_path, ref.local_path)

        coadd = _coadd_from_images(cls, images, outfile_name,
                                   nthreads=nthreads, addbkg=False,
                                   calculate_seeing=False)

        coadd.reference_image = ref
        coadd.target_image = sci
        coadd.header['SEEING'] = sci.header['SEEING']
        coadd.save()

        return coadd


StackedSubtractionFrame = join_model('stackedsubtraction_frames',
                                     MultiEpochSubtraction, SingleEpochSubtraction)


# Detections & Photometry #####################################################

class ObjectWithFlux(models.Base):
    type = sa.Column(sa.Text)

    __tablename__ = 'objectswithflux'
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'base'
    }

    image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                                   ondelete='CASCADE'),
                         index=True)
    image = relationship('CalibratableImage', back_populates='objects',
                         cascade='all')

    # thumbnails = relationship('Thumbnail', cascade='all')

    source_id = sa.Column(sa.Text,
                          sa.ForeignKey('sources.id', ondelete='CASCADE'),
                          index=True)
    source = relationship('Source', cascade='all')

    flux = sa.Column(sa.Float)
    fluxerr = sa.Column(sa.Float)

    @hybrid_property
    def snr(self):
        return self.flux / self.fluxerr


class RealBogus(models.Base):

    __tablename__ = 'realbogus'

    rb_score = sa.Column(sa.Float)
    rb_version = sa.Column(sa.Text)
    detection_id = sa.Column(sa.Integer, sa.ForeignKey('detections.id',
                                                       ondelete='CASCADE'),
                             index=True)
    detection = relationship('Detection', back_populates='rb', cascade='all')



class Detection(ObjectWithFlux, SpatiallyIndexed):
    id = sa.Column(sa.Integer,
                   sa.ForeignKey('objectswithflux.id', ondelete='CASCADE'),
                   primary_key=True)
    __tablename__ = 'detections'
    __mapper_args__ = {'polymorphic_identity': 'detection',
                       'inherit_condition': id == ObjectWithFlux.id}

    x_image = sa.Column(sa.Float)
    y_image = sa.Column(sa.Float)

    elongation = sa.Column(sa.Float)
    a_image = sa.Column(sa.Float)
    b_image = sa.Column(sa.Float)
    fwhm_image = sa.Column(sa.Float)

    flags = sa.Column(sa.Integer)
    imaflags_iso = sa.Column(sa.Integer)
    goodcut = sa.Column(sa.Boolean)
    rb = relationship('RealBogus', cascade='all')

    triggers_alert = sa.Column(sa.Boolean)
    triggers_phot = sa.Column(sa.Boolean)
    alert_ready = sa.Column(sa.Boolean)


    alert = relationship('Alert', cascade='all', uselist=False)

    thumbnails = relationship('Thumbnail', cascade='all')

    @classmethod
    def from_catalog(cls, cat, filter=True):
        result = []

        # TODO: determine if prev dets that are updated should also be returned

        if filter:
            filter_sexcat(cat)

        for row in cat.data:

            if filter and row['GOODCUT'] != 1:
                continue

            # ra and dec are inherited from SpatiallyIndexed
            detection = cls(
                ra=float(row['X_WORLD']), dec=float(row['Y_WORLD']),
                image=cat.image, flux=float(row['FLUX_APER']),
                fluxerr=float(row['FLUXERR_APER']),
                elongation=float(row['ELONGATION']),
                flags=int(row['FLAGS']), imaflags_iso=int(row['IMAFLAGS_ISO']),
                a_image=float(row['A_IMAGE']), b_image=float(row['B_IMAGE']),
                fwhm_image=float(row['FWHM_IMAGE']),
                x_image=float(row['X_IMAGE']), y_image=float(row['Y_IMAGE']),
            )

            rb = RealBogus(rb_score=float(row['rb']), rb_version=BRAAI_MODEL,
                           detection=detection)

            DBSession().add(rb)

            if filter:
                detection.goodcut = True

            result.append(detection)

        return result


class ForcedPhotometry(models.Base):
    id = sa.Column(sa.Integer, primary_key=True)
    __tablename__ = 'forcedphotometry'

    flags = sa.Column(sa.Integer)
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)

    @property
    def mag(self):
        return -2.5 * np.log10(self.flux) + self.image.header['MAGZP'] + \
               self.image.header[APER_KEY]

    @property
    def magerr(self):
        return 1.08573620476 * self.fluxerr / self.flux

    image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id',
                                                   ondelete='CASCADE'),
                         index=True)
    image = relationship('CalibratedImage', back_populates='forced_photometry',
                         cascade='all')

    # thumbnails = relationship('Thumbnail', cascade='all')

    source_id = sa.Column(sa.Text,
                          sa.ForeignKey('sources.id', ondelete='CASCADE'),
                          index=True)
    source = relationship('Source', cascade='all')

    flux = sa.Column(sa.Float)
    fluxerr = sa.Column(sa.Float)

    zp = sa.Column(sa.Float)
    filtercode = sa.Column(sa.Text)
    obsjd = sa.Column(sa.Float)

    @hybrid_property
    def snr(self):
        return self.flux / self.fluxerr


models.Source.thumbnails = relationship('Thumbnail', cascade='all')
models.Source.accumulated_rb = sa.Column(sa.Float)

def images(self, type=CalibratableImage):

    candidates = DBSession().query(type).filter(
        func.q3c_radial_query(type.ra,
                              type.dec,
                              self.ra, self.dec,
                              0.64)
    ).filter(
        func.q3c_poly_query(self.ra, self.dec, type.poly)
    )

    return candidates.all()


def best_detection(self):
    return DBSession().query(
        Detection
    ).join(models.Source).filter(
        models.Source.id == self.id
    ).order_by(
        (Detection.flux / Detection.fluxerr).desc()
    ).first()

models.Source.images = images
models.Source.q3c = Index(
    f'sources_q3c_ang2ipix_idx',
    func.q3c_ang2ipix(
        models.Source.ra,
        models.Source.dec)
)
models.Source.detections = relationship('Detection', cascade='all')
models.Source.forced_photometry = relationship('ForcedPhotometry',
                                               cascade='all')
models.Source.best_detection = property(best_detection)

models.Source.neighbor_info = sa.Column(psql.JSONB)


def unphotometered_images(self):
    subq = DBSession().query(ForcedPhotometry.id,
                             ForcedPhotometry.image_id).filter(
        ForcedPhotometry.source_id == self.id
    ).subquery()

    q = DBSession().query(SingleEpochSubtraction).filter(
        func.q3c_radial_query(SingleEpochSubtraction.ra,
                              SingleEpochSubtraction.dec,
                              self.ra, self.dec,
                              0.64)
    ).filter(
        func.q3c_poly_query(self.ra, self.dec, SingleEpochSubtraction.poly)
    ).outerjoin(
        subq, subq.c.image_id == SingleEpochSubtraction.id
    ).filter(
        subq.c.id == None
    )

    return q

def force_photometry(self, assume_background_subtracted=True):
    out = []
    for i in self.unphotometered_images:
        sci_path = f'/global/cscratch1/sd/dgold/zuds/{i.field:06d}/' \
                   f'c{i.ccdid:02d}/q{i.qid}/{fid_map[i.fid]}/' \
                   f'{i.basename}'

        mask_path = sci_path.replace('.fits', '.mask.fits')
        rms_path = sci_path.replace('.fits', '.rms.fits')

        fp = i.force_photometry(
            self, assume_background_subtracted=assume_background_subtracted,
            use_cutout=True, direct_load={'mask': mask_path, 'sci': sci_path,
                                          'rms': rms_path}
        )
        #r.rms_image.clear()
        #r.mask_image.clear()
        #r.clear()
        out.extend(fp)
    return out

models.Source.unphotometered_images = property(unphotometered_images)
models.Source.force_photometry = force_photometry

from astropy.table import Table
def light_curve(sourceid):
    lc_raw = []

    phot = DBSession().query(
        ForcedPhotometry.obsjd - MJD_TO_JD,
        ForcedPhotometry.filtercode,
        ForcedPhotometry.zp,
        ForcedPhotometry.flux,
        ForcedPhotometry.fluxerr,
        ForcedPhotometry.flags,
        ForcedPhotometry.id
    ).filter(
        ForcedPhotometry.source_id == sourceid
    )

    for photpoint in phot:
        photd = {'mjd': photpoint[0],
                 'filter': 'ztf' + photpoint[1][-1],
                 'zp': photpoint[2],  # ap-correct the ZP
                 'zpsys': 'ab',
                 'flux': photpoint[3],
                 'fluxerr': photpoint[4],
                 'flags': photpoint[5],
                 'lim_mag': -2.5 * np.log10(5 * photpoint[4]) + photpoint[2],
                 'id': photpoint[-1]}
        lc_raw.append(photd)

    return Table(lc_raw)

def lcprop(self):
    return light_curve(self.id)

models.Source.light_curve = lcprop



class FilterRun(models.Base):
    tstart = sa.Column(sa.DateTime)
    tend = sa.Column(sa.DateTime)
    status = sa.Column(sa.Boolean, default=None)
    reason = sa.Column(sa.Text, nullable=True)


class Fit(models.Base):
    success = sa.Column(sa.Boolean)
    message = sa.Column(sa.Text)
    ncall = sa.Column(sa.Integer)
    chisq = sa.Column(sa.Float)
    ndof = sa.Column(sa.Integer)
    param_names = sa.Column(psql.ARRAY(sa.Text))
    parameters = sa.Column(models.NumpyArray)
    vparam_names = sa.Column(psql.ARRAY(sa.Text))
    covariance = sa.Column(models.NumpyArray)
    errors = sa.Column(psql.JSONB)
    nfit = sa.Column(sa.Integer)
    data_mask = sa.Column(psql.ARRAY(sa.Boolean))
    source_id = sa.Column(sa.Text,
                          sa.ForeignKey('sources.id', ondelete='SET NULL'))
    source = relationship('Source')

    @property
    def model(self):
        mod = sncosmo.Model(source='salt2-extended')
        for p, n in zip(self.parameters, self.param_names):
            mod[n] = p
        return mod


models.Source.fits = relationship('Fit', cascade='all')



class DR8(SpatiallyIndexed):
    # hemisphere = sa.Column(sa.Text)

    # __tablename__ = 'dr8'
    # __mapper_args__ = {
    #    'polymorphic_on': hemisphere,
    #    'polymorphic_identity': 'base'
    # }

    # def __repr__(self):
    #    attr_list = [f"{c.name.lower()}={getattr(self, c.name.lower())}"
    #                 for c in self.__table__.columns]
    #    return f"<{type(self).__name__}({', '.join(attr_list)})>"

    release = sa.Column('RELEASE', sa.Integer)
    brickid = sa.Column('BRICKID', sa.Integer)
    brickname = sa.Column('BRICKNAME', sa.Text)
    objid = sa.Column('OBJID', sa.Integer)
    type = sa.Column('TYPE', sa.Text)
    ra = sa.Column('RA', psql.DOUBLE_PRECISION)
    dec = sa.Column('DEC', psql.DOUBLE_PRECISION)
    ra_ivar = sa.Column('RA_IVAR', psql.DOUBLE_PRECISION)
    dec_ivar = sa.Column('DEC_IVAR', psql.DOUBLE_PRECISION)
    ebv = sa.Column('EBV', psql.DOUBLE_PRECISION)
    flux_g = sa.Column('FLUX_G', psql.DOUBLE_PRECISION)
    flux_r = sa.Column('FLUX_R', psql.DOUBLE_PRECISION)
    flux_z = sa.Column('FLUX_Z', psql.DOUBLE_PRECISION)
    flux_w1 = sa.Column('FLUX_W1', psql.DOUBLE_PRECISION)
    flux_w2 = sa.Column('FLUX_W2', psql.DOUBLE_PRECISION)
    flux_w3 = sa.Column('FLUX_W3', psql.DOUBLE_PRECISION)
    flux_w4 = sa.Column('FLUX_W4', psql.DOUBLE_PRECISION)
    flux_ivar_g = sa.Column('FLUX_IVAR_G', psql.DOUBLE_PRECISION)
    flux_ivar_r = sa.Column('FLUX_IVAR_R', psql.DOUBLE_PRECISION)
    flux_ivar_z = sa.Column('FLUX_IVAR_Z', psql.DOUBLE_PRECISION)
    flux_ivar_w1 = sa.Column('FLUX_IVAR_W1', psql.DOUBLE_PRECISION)
    flux_ivar_w2 = sa.Column('FLUX_IVAR_W2', psql.DOUBLE_PRECISION)
    flux_ivar_w3 = sa.Column('FLUX_IVAR_W3', psql.DOUBLE_PRECISION)
    flux_ivar_w4 = sa.Column('FLUX_IVAR_W4', psql.DOUBLE_PRECISION)
    mw_transmission_g = sa.Column('MW_TRANSMISSION_G', psql.DOUBLE_PRECISION)
    mw_transmission_r = sa.Column('MW_TRANSMISSION_R', psql.DOUBLE_PRECISION)
    mw_transmission_z = sa.Column('MW_TRANSMISSION_Z', psql.DOUBLE_PRECISION)
    mw_transmission_w1 = sa.Column('MW_TRANSMISSION_W1', psql.DOUBLE_PRECISION)
    mw_transmission_w2 = sa.Column('MW_TRANSMISSION_W2', psql.DOUBLE_PRECISION)
    mw_transmission_w3 = sa.Column('MW_TRANSMISSION_W3', psql.DOUBLE_PRECISION)
    mw_transmission_w4 = sa.Column('MW_TRANSMISSION_W4', psql.DOUBLE_PRECISION)
    nobs_g = sa.Column('NOBS_G', sa.Integer)
    nobs_r = sa.Column('NOBS_R', sa.Integer)
    nobs_z = sa.Column('NOBS_Z', sa.Integer)
    nobs_w1 = sa.Column('NOBS_W1', sa.Integer)
    nobs_w2 = sa.Column('NOBS_W2', sa.Integer)
    nobs_w3 = sa.Column('NOBS_W3', sa.Integer)
    nobs_w4 = sa.Column('NOBS_W4', sa.Integer)
    rchisq_g = sa.Column('RCHISQ_G', psql.DOUBLE_PRECISION)
    rchisq_r = sa.Column('RCHISQ_R', psql.DOUBLE_PRECISION)
    rchisq_z = sa.Column('RCHISQ_Z', psql.DOUBLE_PRECISION)
    rchisq_w1 = sa.Column('RCHISQ_W1', psql.DOUBLE_PRECISION)
    rchisq_w2 = sa.Column('RCHISQ_W2', psql.DOUBLE_PRECISION)
    rchisq_w3 = sa.Column('RCHISQ_W3', psql.DOUBLE_PRECISION)
    rchisq_w4 = sa.Column('RCHISQ_W4', psql.DOUBLE_PRECISION)
    fracflux_g = sa.Column('FRACFLUX_G', psql.DOUBLE_PRECISION)
    fracflux_r = sa.Column('FRACFLUX_R', psql.DOUBLE_PRECISION)
    fracflux_z = sa.Column('FRACFLUX_Z', psql.DOUBLE_PRECISION)
    fracflux_w1 = sa.Column('FRACFLUX_W1', psql.DOUBLE_PRECISION)
    fracflux_w2 = sa.Column('FRACFLUX_W2', psql.DOUBLE_PRECISION)
    fracflux_w3 = sa.Column('FRACFLUX_W3', psql.DOUBLE_PRECISION)
    fracflux_w4 = sa.Column('FRACFLUX_W4', psql.DOUBLE_PRECISION)
    fracmasked_g = sa.Column('FRACMASKED_G', psql.DOUBLE_PRECISION)
    fracmasked_r = sa.Column('FRACMASKED_R', psql.DOUBLE_PRECISION)
    fracmasked_z = sa.Column('FRACMASKED_Z', psql.DOUBLE_PRECISION)
    fracin_g = sa.Column('FRACIN_G', psql.DOUBLE_PRECISION)
    fracin_r = sa.Column('FRACIN_R', psql.DOUBLE_PRECISION)
    fracin_z = sa.Column('FRACIN_Z', psql.DOUBLE_PRECISION)
    anymask_g = sa.Column('ANYMASK_G', sa.Integer)
    anymask_r = sa.Column('ANYMASK_R', sa.Integer)
    anymask_z = sa.Column('ANYMASK_Z', sa.Integer)
    allmask_g = sa.Column('ALLMASK_G', sa.Integer)
    allmask_r = sa.Column('ALLMASK_R', sa.Integer)
    allmask_z = sa.Column('ALLMASK_Z', sa.Integer)
    wisemask_w1 = sa.Column('WISEMASK_W1', sa.Integer)
    wisemask_w2 = sa.Column('WISEMASK_W2', sa.Integer)
    psfsize_g = sa.Column('PSFSIZE_G', psql.DOUBLE_PRECISION)
    psfsize_r = sa.Column('PSFSIZE_R', psql.DOUBLE_PRECISION)
    psfsize_z = sa.Column('PSFSIZE_Z', psql.DOUBLE_PRECISION)
    psfdepth_g = sa.Column('PSFDEPTH_G', psql.DOUBLE_PRECISION)
    psfdepth_r = sa.Column('PSFDEPTH_R', psql.DOUBLE_PRECISION)
    psfdepth_z = sa.Column('PSFDEPTH_Z', psql.DOUBLE_PRECISION)
    galdepth_g = sa.Column('GALDEPTH_G', psql.DOUBLE_PRECISION)
    galdepth_r = sa.Column('GALDEPTH_R', psql.DOUBLE_PRECISION)
    galdepth_z = sa.Column('GALDEPTH_Z', psql.DOUBLE_PRECISION)
    psfdepth_w1 = sa.Column('PSFDEPTH_W1', psql.DOUBLE_PRECISION)
    psfdepth_w2 = sa.Column('PSFDEPTH_W2', psql.DOUBLE_PRECISION)
    wise_coadd_id = sa.Column('WISE_COADD_ID', sa.Text)
    fracdev = sa.Column('FRACDEV', psql.DOUBLE_PRECISION)
    fracdev_ivar = sa.Column('FRACDEV_IVAR', psql.DOUBLE_PRECISION)
    shapedev_r = sa.Column('SHAPEDEV_R', psql.DOUBLE_PRECISION)
    shapedev_r_ivar = sa.Column('SHAPEDEV_R_IVAR', psql.DOUBLE_PRECISION)
    shapedev_e1 = sa.Column('SHAPEDEV_E1', psql.DOUBLE_PRECISION)
    shapedev_e1_ivar = sa.Column('SHAPEDEV_E1_IVAR', psql.DOUBLE_PRECISION)
    shapedev_e2 = sa.Column('SHAPEDEV_E2', psql.DOUBLE_PRECISION)
    shapedev_e2_ivar = sa.Column('SHAPEDEV_E2_IVAR', psql.DOUBLE_PRECISION)
    shapeexp_r = sa.Column('SHAPEEXP_R', psql.DOUBLE_PRECISION)
    shapeexp_r_ivar = sa.Column('SHAPEEXP_R_IVAR', psql.DOUBLE_PRECISION)
    shapeexp_e1 = sa.Column('SHAPEEXP_E1', psql.DOUBLE_PRECISION)
    shapeexp_e1_ivar = sa.Column('SHAPEEXP_E1_IVAR', psql.DOUBLE_PRECISION)
    shapeexp_e2 = sa.Column('SHAPEEXP_E2', psql.DOUBLE_PRECISION)
    shapeexp_e2_ivar = sa.Column('SHAPEEXP_E2_IVAR', psql.DOUBLE_PRECISION)
    fiberflux_g = sa.Column('FIBERFLUX_G', psql.DOUBLE_PRECISION)
    fiberflux_r = sa.Column('FIBERFLUX_R', psql.DOUBLE_PRECISION)
    fiberflux_z = sa.Column('FIBERFLUX_Z', psql.DOUBLE_PRECISION)
    fibertotflux_g = sa.Column('FIBERTOTFLUX_G', psql.DOUBLE_PRECISION)
    fibertotflux_r = sa.Column('FIBERTOTFLUX_R', psql.DOUBLE_PRECISION)
    fibertotflux_z = sa.Column('FIBERTOTFLUX_Z', psql.DOUBLE_PRECISION)
    ref_cat = sa.Column('REF_CAT', sa.Text)
    ref_id = sa.Column('REF_ID', sa.Integer)
    ref_epoch = sa.Column('REF_EPOCH', psql.DOUBLE_PRECISION)
    gaia_phot_g_mean_mag = sa.Column('GAIA_PHOT_G_MEAN_MAG',
                                     psql.DOUBLE_PRECISION)
    gaia_phot_g_mean_flux_over_error = sa.Column(
        'GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', psql.DOUBLE_PRECISION)
    gaia_phot_bp_mean_mag = sa.Column('GAIA_PHOT_BP_MEAN_MAG',
                                      psql.DOUBLE_PRECISION)
    gaia_phot_bp_mean_flux_over_error = sa.Column(
        'GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', psql.DOUBLE_PRECISION)
    gaia_phot_rp_mean_mag = sa.Column('GAIA_PHOT_RP_MEAN_MAG',
                                      psql.DOUBLE_PRECISION)
    gaia_phot_rp_mean_flux_over_error = sa.Column(
        'GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', psql.DOUBLE_PRECISION)
    gaia_astrometric_excess_noise = sa.Column('GAIA_ASTROMETRIC_EXCESS_NOISE',
                                              psql.DOUBLE_PRECISION)
    gaia_duplicated_source = sa.Column('GAIA_DUPLICATED_SOURCE', sa.Boolean)
    gaia_phot_bp_rp_excess_factor = sa.Column('GAIA_PHOT_BP_RP_EXCESS_FACTOR',
                                              psql.DOUBLE_PRECISION)
    gaia_astrometric_sigma5d_max = sa.Column('GAIA_ASTROMETRIC_SIGMA5D_MAX',
                                             psql.DOUBLE_PRECISION)
    gaia_astrometric_params_solved = sa.Column('GAIA_ASTROMETRIC_PARAMS_SOLVED',
                                               sa.Integer)
    parallax = sa.Column('PARALLAX', psql.DOUBLE_PRECISION)
    parallax_ivar = sa.Column('PARALLAX_IVAR', psql.DOUBLE_PRECISION)
    pmra = sa.Column('PMRA', psql.DOUBLE_PRECISION)
    pmra_ivar = sa.Column('PMRA_IVAR', psql.DOUBLE_PRECISION)
    pmdec = sa.Column('PMDEC', psql.DOUBLE_PRECISION)
    pmdec_ivar = sa.Column('PMDEC_IVAR', psql.DOUBLE_PRECISION)
    maskbits = sa.Column('MASKBITS', sa.Integer)
    z_phot_mean = sa.Column('z_phot_mean', psql.DOUBLE_PRECISION)
    z_phot_median = sa.Column('z_phot_median', psql.DOUBLE_PRECISION)
    z_phot_std = sa.Column('z_phot_std', psql.DOUBLE_PRECISION)
    z_phot_l68 = sa.Column('z_phot_l68', psql.DOUBLE_PRECISION)
    z_phot_u68 = sa.Column('z_phot_u68', psql.DOUBLE_PRECISION)
    z_phot_l95 = sa.Column('z_phot_l95', psql.DOUBLE_PRECISION)
    z_phot_u95 = sa.Column('z_phot_u95', psql.DOUBLE_PRECISION)
    z_spec = sa.Column('z_spec', psql.DOUBLE_PRECISION)
    survey = sa.Column('survey', sa.Text)
    training = sa.Column('training', sa.Boolean)

    @hybrid_property
    def gmag(self):
        return -2.5 * np.log10(self.flux_g) + 22.5

    @gmag.expression
    def gmag(self):
        return -2.5 * sa.func.log(self.flux_g) + 22.5

    @hybrid_property
    def rmag(self):
        return -2.5 * np.log10(self.flux_r) + 22.5

    @rmag.expression
    def rmag(self):
        return -2.5 * sa.func.log(self.flux_r) + 22.5

    @hybrid_property
    def w1mag(self):
        return -2.5 * np.log10(self.flux_w1) + 22.5

    @w1mag.expression
    def w1mag(self):
        return -2.5 * sa.func.log(self.flux_w1) + 22.5


class DR8North(models.Base, DR8):
    # id = sa.Column(sa.Integer, sa.ForeignKey('dr8.id', ondelete='CASCADE'),
    # primary_key=True)
    __tablename__ = 'dr8_north'
    # __mapper_args__ = {'polymorphic_identity': 'n'}

    # @declared_attr
    # def __table_args__(cls):
    #    return tuple()


class DR8South(models.Base, DR8):
    # id = sa.Column(sa.Integer, sa.ForeignKey('dr8.id', ondelete='CASCADE'),
    # primary_key=True)
    __tablename__ = 'dr8_south'
    # __mapper_args__ = {'polymorphic_identity': 's'}

    # @declared_attr
    # def __table_args__(cls):
    #    return tuple()

init_db()

from sqlalchemy import event

models.Thumbnail.image = relationship('CalibratableImage',
                     cascade='all',
                     back_populates='thumbnails',
                     foreign_keys=[models.Thumbnail.image_id])


models.Thumbnail.source = relationship(
    'Source',
    cascade='all',
    back_populates='thumbnails',
    foreign_keys=[models.Thumbnail.source_id]
)

models.Thumbnail.detection = relationship(
    'Detection',
    cascade='all',
    back_populates='thumbnails',
    foreign_keys=[models.Thumbnail.detection_id]
)


def stamp_to_array(self):
    if self.bytes is None:
        raise ValueError('Cannot coerce array from empty bytes attribute')
    fitsbuf = io.BytesIO(gzip.decompress(self.bytes))
    array = np.flipud(fits.open(fitsbuf)[0].data)
    return array

models.Thumbnail.array = property(stamp_to_array)


@event.listens_for(DBSession(), 'before_flush')
def bump_modified(session, flush_context, instances):
    for object in session.dirty:
        if isinstance(object, models.Base) and session.is_modified(object):
            object.modified = sa.func.now()

class CLU(models.Base):

    __tablename__ = 'clu_20190625'

    cluid = sa.Column(sa.Integer, primary_key=True)
    id_other = sa.Column(sa.Text)
    name = sa.Column(sa.Text)
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)
    dm = sa.Column(sa.Float)
    dm_method = sa.Column(sa.Text)
    distmpc = sa.Column(sa.Float)
    dm_kin = sa.Column(sa.Float)
    z = sa.Column(sa.Float)
    zerr = sa.Column(sa.Float)
    a = sa.Column(sa.Float)
    b2a = sa.Column(sa.Float)
    pa = sa.Column(sa.Float)
    type_ned = sa.Column(sa.Text)

from copy import deepcopy
def print_time(start, stop, detection, step):
    print(f'took {stop-start:.2f} sec to do {step} for {detection.id}', flush=True)

class Alert(models.Base):
    alert = sa.Column(psql.JSONB)
    creation_index = sa.Index('created_at_index', 'created_at')

    cutoutscience_id = sa.Column(sa.Integer, sa.ForeignKey(
        'thumbnails.id', ondelete='SET NULL'
    ))

    cutouttemplate_id = sa.Column(sa.Integer, sa.ForeignKey(
        'thumbnails.id', ondelete='SET NULL'
    ))

    cutoutdifference_id = sa.Column(sa.Integer, sa.ForeignKey(
        'thumbnails.id', ondelete='SET NULL'
    ))

    detection_id = sa.Column(sa.Integer, sa.ForeignKey(
        'detections.id', ondelete='SET NULL'
    ), index=True)

    cutoutscience = relationship('Thumbnail', cascade='all',
                                 foreign_keys=[cutoutscience_id])
    cutouttemplate = relationship('Thumbnail', cascade='all',
                                  foreign_keys=[cutouttemplate_id])
    cutoutdifference = relationship('Thumbnail', cascade='all',
                                    foreign_keys=[cutoutdifference_id])

    detection = relationship('Detection', cascade='all',
                             foreign_keys=[detection_id])

    sent = sa.Column(sa.Boolean, default=False)

    def to_dict(self):
        base = deepcopy(self.alert)
        base['cutoutScience'] = self.cutoutscience.bytes
        base['cutoutTemplate'] = self.cutouttemplate.bytes
        base['cutoutDifference'] = self.cutoutdifference.bytes
        return base

    @classmethod
    def from_detection(cls, detection):
        obj = cls()

        if detection.source is None:
            raise ValueError('Cannot issue an alert for a detection that is '
                             f'not associated with a source ({detection.id})')

        # prepare the JSONB
        alert = dict()
        alert['objectId'] = detection.source.id
        alert['candid'] = detection.id
        alert['schemavsn'] = '0.4'
        alert['publisher'] = 'ZUDS/NERSC'

        # do a bunch of cross matches to initially populate the candidate
        # subschema
        start = time.time()
        candidate = xmatch(detection.ra, detection.dec, detection.source.id)
        alert['candidate'] = candidate
        stop = time.time()
        print_time(start, stop, detection, 'xmatch')


        # indicate whether this alert is generated based on a stack detection
        #  or a single epoch detection
        start = time.time()
        alert_type = 'single' if isinstance(
            detection.image, SingleEpochSubtraction
        ) else 'stack'
        candidate['alert_type'] = alert_type
        stop = time.time()
        print_time(start, stop, detection , 'alert_type')


        # add some basic info about the image this was found on and metadata
        start = time.time()
        candidate['fid'] = detection.image.fid
        candidate['pid'] = detection.image.id

        if isinstance(detection.image, SingleEpochSubtraction):
            candidate['programpi'] = detection.image.target_image.header['PROGRMPI']
            candidate['programid'] = detection.image.target_image.header['PROGRMID']
        else:
            candidate['programpi'] = \
                detection.image.target_image.input_images[0].header['PROGRMPI']
            candidate['programid'] = \
                detection.image.target_image.input_images[0].header['PROGRMID']
        candidate['pdiffimfilename'] = detection.image.basename
        candidate['candid'] = detection.id
        candidate['isdiffpos'] = 't'
        candidate['field'] = detection.image.field
        candidate['ra'] = detection.ra
        candidate['dec'] = detection.dec
        candidate['rcid'] = (detection.image.ccdid - 1) * 4 + (
            detection.image.qid - 1)

        # shape and position information
        candidate['aimage'] = detection.a_image
        candidate['bimage'] = detection.b_image
        candidate['elong'] = detection.elongation
        candidate['fwhm'] = detection.fwhm_image
        candidate['aimagerat'] = detection.a_image / detection.fwhm_image
        candidate['bimagerat'] = detection.b_image / detection.fwhm_image
        candidate['xpos'] = detection.x_image
        candidate['ypos'] = detection.y_image

        # flux information
        candidate['snr'] = detection.snr

        # machine learning
        candidate['drb'] = detection.rb[0].rb_score
        candidate['drbversion'] = detection.rb[0].rb_version
        stop = time.time()
        print_time(start, stop, detection, 'basic')

        # information about the reference images

        start = time.time()

        refmin, refmax, refcount = DBSession().query(
            sa.func.min(ScienceImage.obsjd) - MJD_TO_JD,
            sa.func.max(ScienceImage.obsjd) - MJD_TO_JD,
            sa.func.count(ScienceImage.id)
        ).select_from(ScienceImage.__table__).join(
            CoaddImage, ScienceImage.id == CoaddImage.calibratableimage_id
        ).filter(
            CoaddImage.coadd_id == detection.image.reference_image_id
        ).first()
        stop = time.time()
        print_time(start, stop, detection, 'get refimgs')

        start = time.time()

        if alert_type == 'single':
            candidate['jd'] = detection.image.mjd + MJD_TO_JD
            candidate['nid'] = detection.image.target_image.nid
            candidate['diffmaglim'] = detection.image.target_image.maglimit
            candidate['exptime'] = detection.image.target_image.exptime
            mjdcut = detection.image.mjd
        else:
            stackimgs = sorted(
                detection.image.target_image.input_images,
                key=lambda i: i.mjd
            )
            candidate['jdstartstack'] = stackimgs[0].mjd + MJD_TO_JD
            candidate['jdendstack'] = stackimgs[-1].mjd + MJD_TO_JD
            candidate['jdmed'] = np.median([i.mjd + MJD_TO_JD for i in stackimgs])
            candidate['nframesstack'] = len(stackimgs)
            candidate['exptime'] = np.sum([i.exptime for i in stackimgs])
            mjdcut = stackimgs[-1].mjd

        # add half a second to mjdcut to ensure expected behavior in
        # floating point comparisons
        mjdcut += 0.5 / 3600. / 24.

        stop = time.time()
        print_time(start, stop, detection, 'histstats')

        start = time.time()

        # calculate the detection history

        if alert_type == 'single':
            mymjd = detection.image.mjd
        else:
            mymjd = detection.image.target_image.max_mjd

        # get dates of single epoch detections
        singledates = DBSession().query(ScienceImage.obsjd - MJD_TO_JD).select_from(
            ScienceImage.__table__
        ).join(
            SingleEpochSubtraction.__table__, SingleEpochSubtraction.target_image_id == ScienceImage.id
        ).join(
            ObjectWithFlux.__table__, ObjectWithFlux.image_id == SingleEpochSubtraction.id
        ).filter(
            ObjectWithFlux.source_id == detection.source.id
        ).all()
        singledates = list(sorted([date[0] for date in singledates if date[0] < mymjd]))

        multidates = DBSession().query(sa.func.min(ScienceImage.obsjd) - MJD_TO_JD,
                                       sa.func.max(ScienceImage.obsjd) - MJD_TO_JD
        ).select_from(
            ScienceImage.__table__
        ).join(
            CoaddImage, CoaddImage.calibratableimage_id == ScienceImage.id
        ).join(
            ScienceCoadd.__table__, ScienceCoadd.id == CoaddImage.coadd_id
        ).join(
            MultiEpochSubtraction.__table__, MultiEpochSubtraction.target_image_id == ScienceCoadd.id
        ).join(
            ObjectWithFlux.__table__, ObjectWithFlux.image_id == MultiEpochSubtraction.id
        ).filter(
            ObjectWithFlux.source_id == detection.source.id
        ).group_by(ScienceCoadd.id)

        multidates = list(sorted([date for date in multidates if date[0] < mymjd],
                                 key=lambda d: d[1]))

        candidate['ndethist_single'] = len(singledates)
        candidate['ndethist_stack'] = len(multidates)

        stop = time.time()
        print_time(start, stop, detection, 'prevdets')

        start = time.time()

        if len(singledates) > 0:
            starthist = singledates[0] + MJD_TO_JD
            endhist = singledates[-1] + MJD_TO_JD
            candidate['jdstarthist_single'] = starthist
            candidate['jdendhist_single'] = endhist
        else:
            candidate['jdstarthist_single'] = None
            candidate['jdendhist_single'] = None

        if len(multidates) > 0:
            starthist = multidates[0][0] + MJD_TO_JD
            endhist = multidates[-1][1] + MJD_TO_JD
            candidate['jdstarthist_stack'] = starthist
            candidate['jdendhist_stack'] = endhist
        else:
            candidate['jdstarthist_stack'] = None
            candidate['jdendhist_stack'] = None

        stop = time.time()
        print_time(start, stop, detection, 'jdstarthist')

        start = time.time()
        # make the light curve
        lc = detection.source.light_curve()
        stop = time.time()
        print_time(start, stop, detection, 'light_curve')

        if len(lc) == 0:
            raise RuntimeError('Cannot issue an alert for an object with no '
                               'light curve, please rerun forced photometry '
                               'to update the light curve of this object:' 
                               f'"{detection.source.id}".')

        lc = lc[lc['mjd'] <= mjdcut]
        alert['light_curve'] = lc.to_pandas().to_dict(orient='records')

        candidate['jdstartref'] = refmin
        candidate['jdendref'] = refmax
        candidate['nframesref'] = refcount

        start = time.time()
        # now do the cutouts
        for stamp in detection.thumbnails:
            if stamp.type == 'ref':
                obj.cutouttemplate = stamp
            elif stamp.type == 'new':
                obj.cutoutscience = stamp
            elif stamp.type == 'sub':
                obj.cutoutdifference = stamp

        stop = time.time()
        print_time(start, stop, detection, 'cutouts')

        obj.alert = alert

        # this is to prevent detections from being re-inserted, triggering
        # a unique key violation

        obj.detection_id = detection.id
        return obj



class Job(models.Base):
    status = sa.Column(sa.Text, index=True, default='unsubmitted',
                       nullable=False)
    images = relationship('CalibratableImage', secondary='job_images')
    slurm_id = sa.Column(sa.Text, index=True, nullable=False)

JobImage = join_model('job_images', Job, CalibratableImage)


class ForcePhotJob(models.Base):
    status = sa.Column(sa.Text, index=True, default='unsubmitted',
                       nullable=False)
    slurm_id = sa.Column(sa.Text, index=True, nullable=False)
    detection_file = sa.Column(sa.Text)
    output_file = sa.Column(sa.Text)

class AlertJob(models.Base):
    status = sa.Column(sa.Text, index=True, default='unsubmitted',
                       nullable=False)
    slurm_id = sa.Column(sa.Text, index=True, nullable=False)


class FailedSubtraction(models.Base):

    target_image_id = sa.Column(sa.Integer, sa.ForeignKey(
        'calibratableimages.id'
    ), index=True)
    reference_image_id = sa.Column(sa.Integer, sa.ForeignKey(
        'referenceimages.id'
    ), index=True)

    target_image = relationship('CalibratableImage', cascade='all',
                                foreign_keys=[target_image_id])
    reference_image = relationship('ReferenceImage', cascade='all',
                                   foreign_keys=[reference_image_id])

    reason = sa.Column(sa.Text)

