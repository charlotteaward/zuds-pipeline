import os
from skyportal.models import DBSession
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.exc import UnboundExecutionError
from sqlalchemy.ext.hybrid import hybrid_property

from skyportal import models
from skyportal.models import DBSession, join_model
from skyportal.model_util import create_tables, drop_tables

from .file import File
from .secrets import get_secret
from .utils import fid_map

__all__ = ['DBSession', 'create_tables', 'drop_tables',
           'Base', 'init_db', 'join_model', 'ZTFFile']


def init_db(timeout=None):

    username = get_secret('db_username')
    password = get_secret('db_password')
    port = get_secret('db_port')
    host = get_secret('db_host')
    dbname = get_secret('db_name')

    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(username, password or '', host or '', port or '', dbname)

    kwargs = {}
    if timeout is not None:
        kwargs['connect_args'] = {"options": f"-c statement_timeout={timeout}"}

    conn = sa.create_engine(url, client_encoding='utf8', **kwargs)

    DBSession.configure(bind=conn)
    models.Base.metadata.bind = conn


def model_representation(o):
    """String representation of sqlalchemy objects."""
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

# Automatically update the `modified` attribute of Base
# when objects are updated.
@event.listens_for(DBSession(), 'before_flush')
def bump_modified(session, flush_context, instances):
    for object in session.dirty:
        if isinstance(object, models.Base) and session.is_modified(object):
            object.modified = sa.func.now()


Base = models.Base


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

        try:
            obj = DBSession().query(cls).filter(
                cls.basename == basename
            ).first()
        except UnboundExecutionError:
            # Running without a database
            obj = None

        if obj is not None:
            obj.clear()  # get a fresh copy

        if hasattr(obj, 'mask_image'):
            if obj.mask_image is not None:
                obj.mask_image.clear()

        if hasattr(obj, 'catalog'):
            if obj.catalog is not None:
                obj.catalog.clear()

        return obj

    @property
    def relname(self):
        return f'{self.field:06d}/' \
               f'c{self.ccdid:02d}/' \
               f'q{self.qid}/' \
               f'{fid_map[self.fid]}/' \
               f'{self.basename}'

    @hybrid_property
    def relname_hybrid(self):
        return sa.func.format(
            '%s/c%s/q%s/%s/%s',
            sa.func.lpad(sa.func.cast(self.field, sa.Text), 6, '0'),
            sa.func.lpad(sa.func.cast(self.ccdid, sa.Text), 2, '0'),
            self.qid,
            sa.case([
                (self.fid == 1, 'zg'),
                (self.fid == 2, 'zr'),
                (self.fid == 3, 'zi')
            ]),
            self.basename
        )