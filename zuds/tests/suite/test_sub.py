import zuds
import numpy as np
import requests
import pytest
import os
import uuid

stampcent = np.array([[-11.565292 , -16.885056 ,   3.7509918,  -3.0413055,  -2.3746796,
        -28.3004   ],
       [ -9.599594 ,  -1.142746 ,  16.329361 ,  15.714432 ,   7.308548 ,
         -6.655197 ],
       [ 21.798233 ,  22.084488 ,  12.057449 , -11.082443 , -32.319443 ,
         10.269241 ],
       [ 10.716263 ,   1.418045 ,   6.831436 , -29.350693 ,  14.179764 ,
         25.866104 ],
       [  8.922256 ,  24.062637 ,  -5.965805 ,  47.361404 , -10.360397 ,
         -7.7291107],
       [-12.233261 ,  20.014725 , -17.911667 , -14.06749  , -36.002853 ,
         -7.4232025]])

@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_sub(sci_image_data_20200604, refimg_data_first2_imgs):
    _ = sci_image_data_20200604.weight_image
    sub = zuds.SingleEpochSubtraction.from_images(sci_image_data_20200604,
                                                  refimg_data_first2_imgs,
                                                  nreg_side=1,
                                                  hotpants_kws={'ko': 0,
                                                                'bgo': 0})
    naxis1, naxis2 = sub.header['NAXIS1'], sub.header['NAXIS2']
    stamp = sub.data[naxis1 // 2 - 3:naxis1 // 2 + 3,
            naxis2 // 2 - 3:naxis2 // 2 + 3]
    assert naxis1 == 495
    assert naxis2 == 495
    np.testing.assert_allclose(stamp, stampcent)
    assert sub.reference_image is refimg_data_first2_imgs
    assert sub.target_image is sci_image_data_20200604


def test_multi_epoch_sub(sci_image_data_20200601, sci_image_data_20200604,
                         refimg_data_first2_imgs):

    outdir = os.path.dirname(sci_image_data_20200601.local_path)
    outname = os.path.join(outdir, f'{uuid.uuid4().hex}.fits')

    coadd = zuds.ScienceCoadd.from_images([sci_image_data_20200604,
                                           sci_image_data_20200601],
                                          outname)
    coadd.binleft = '2020-06-01'
    coadd.binright = '2020-06-04'

    zuds.DBSession().add(coadd)

    for a in coadd.input_images:
        se_sub = zuds.SingleEpochSubtraction.from_images(
            a, refimg_data_first2_imgs
        )

        zuds.DBSession().add(se_sub)

    final = zuds.MultiEpochSubtraction.from_images(
        coadd, refimg_data_first2_imgs, force_map_subs=False,
    )
    zuds.DBSession().add(final)
    zuds.DBSession().commit()
