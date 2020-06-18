import uuid
import shutil
from pathlib import Path
import subprocess


from .utils import initialize_directory
from .constants import BKG_BOX_SIZE
from .mask import MaskImageBase, MaskImage

__all__ = ['prepare_swarp_sci', 'prepare_swarp_mask', 'prepare_swarp_align',
           'run_align']


CONF_DIR = Path(__file__).parent / 'astromatic/makecoadd'
SCI_CONF = CONF_DIR / 'default.swarp'
MSK_CONF = CONF_DIR / 'mask.swarp'


def prepare_swarp_sci(images, outname, directory, swarp_kws=None,
                      swarp_zp_key='MAGZP'):

    conf = SCI_CONF
    initialize_directory(directory)

    impaths = [im.local_path for im in images]

    # normalize all images to the same zeropoint
    for im, path in zip(images, impaths):
        if swarp_zp_key in im.header:
            fluxscale = 10**(-0.4 * (im.header[swarp_zp_key] - 25.))
            im.header['FLXSCALE'] = fluxscale
            im.header_comments['FLXSCALE'] = 'Flux scale factor for coadd / DG'
            im.header['FLXSCLZP'] = 25.
            im.header_comments['FLXSCLZP'] = 'FLXSCALE equivalent ZP / DG'
            opath = im.local_path
            im.map_to_local_file(path)
            im.save()
            im.map_to_local_file(opath)

    # if weight images do not exist yet, write them to temporary
    # directory
    wgtpaths = []
    for image in images:
        if not image.weight_image.ismapped:
            wgtpath = f"{directory / image.basename.replace('.fits', '.weight.fits')}"
            image.weight_image.map_to_local_file(wgtpath)
            image.weight_image.save()
        else:
            wgtpath = image.weight_image.local_path
        wgtpaths.append(wgtpath)

    # need to write the images to a list, to avoid shell commands that are too
    # long and trigger SIGSEGV
    inlist = directory / 'images.in'
    with open(inlist, 'w') as f:
        for p in impaths:
            f.write(f'{p}\n')

    inweight = directory / 'weight.in'
    with open(inweight, 'w') as f:
        for p in wgtpaths:
            f.write(f'{p}\n')

    # get the output weight image in string form
    wgtout = outname.replace('.fits', '.weight.fits')

    syscall = f'swarp -c {conf} @{inlist} ' \
              f'-BACK_SIZE {BKG_BOX_SIZE} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHT_IMAGE @{inweight} ' \
              f'-WEIGHTOUT_NAME {wgtout} '

    if swarp_kws is not None:
        for kw in swarp_kws:
            syscall += f'-{kw.upper()} {swarp_kws[kw]} '

    return syscall


def prepare_swarp_mask(masks, outname, mskoutweightname, directory,
                       swarp_kws=None):


    conf = MSK_CONF
    initialize_directory(directory)

    # get the images in string form
    allims = ' '.join([c.local_path for c in masks])

    syscall = f'swarp -c {conf} {allims} ' \
              f'-SUBTRACT_BACK N ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHTOUT_NAME {mskoutweightname} '

    if swarp_kws is not None:
        for kw in swarp_kws:
            syscall += f'-{kw.upper()} {swarp_kws[kw]} '

    return syscall


def prepare_swarp_align(image, other, directory, nthreads=1,
                        persist_aligned=False):

    from astropy.wcs import WCS
    conf = SCI_CONF
    shutil.copy(image.local_path, directory)
    impath = str(directory / image.basename)
    align_header = other.astropy_header

    # now get the WCS keys to align the header to
    head = WCS(align_header).to_header(relax=True)

    # and write the results to a file that swarp will read
    extension = f'_aligned_to_{other.basename[:-5]}.remap'

    if persist_aligned:
        outname = image.local_path.replace('.fits', f'{extension}.fits')
    else:
        outname = impath.replace('.fits', f'{extension}.fits')
    headpath = impath.replace('.fits', f'{extension}.head')

    with open(headpath, 'w') as f:
        for card in align_header.cards:
            if card.keyword.startswith('NAXIS'):
                f.write(f'{card.image}\n')
        for card in head.cards:
            f.write(f'{card.image}\n')

    # make a random file for the weightmap -> we dont want to use it
    weightname = directory / image.basename.replace(
        '.fits',
        f'{extension}.weight.fits'
    )

    combtype = 'OR' if isinstance(image, MaskImageBase) else 'CLIPPED'

    syscall = f'swarp -c {conf} {impath} ' \
              f'-BACK_SIZE {BKG_BOX_SIZE} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-NTHREADS {nthreads} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-SUBTRACT_BACK N ' \
              f'-WEIGHTOUT_NAME {weightname} ' \
              f'-WEIGHT_TYPE NONE ' \
              f'-COMBINE_TYPE {combtype} '

    return syscall, outname, weightname


def run_align(image, other, tmpdir='/tmp',
              nthreads=1, persist_aligned=False):

    from .image import FITSImage

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command, outname, outweight = prepare_swarp_align(
        image, other,
        directory,
        nthreads=nthreads,
        persist_aligned=persist_aligned
    )

    # run swarp
    while True:
        try:
            subprocess.check_call(command.split())
        except OSError as e:
            if e.errno == 14:
                continue
            else:
                raise e
        else:
            break

    restype = MaskImageBase if isinstance(image, MaskImage) else FITSImage

    result = restype.from_file(outname)
    result.parent_image = image
    weightimage = FITSImage.from_file(outweight)

    if isinstance(image, MaskImage):
        result.update_from_weight_map(weightimage)

    # load everything into memory and unmap if the disk file is going to be
    # deleted
    if not persist_aligned:
        result.load()

        # unmap the object from disk, but preserve the loaded attrs.
        del result._path

    # clean up the swarp working dir
    shutil.rmtree(directory)

    return result

