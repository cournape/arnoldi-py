import numpy as np
import numpy.linalg as nlin

from arnoldi.matrices import laplace, laplace_eigen, mark


class TestMatrices:
    def test_mark_2(self):
        ## Given
        r_mark = np.array(
            [[0. , 1. , 1. ],
             [0.5, 0. , 0. ],
             [0.5, 0. , 0. ]]
        )

        # When / Then
        np.testing.assert_array_almost_equal(mark(2).todense(), r_mark)

    def test_mark_3(self):
        ## Given
        r_mark = np.array(
            [[0.  , 0.5 , 0.  , 0.5 , 0.  , 0.  ],
             [0.5 , 0.  , 1.  , 0.  , 0.5 , 0.  ],
             [0.  , 0.25, 0.  , 0.  , 0.  , 0.  ],
             [0.5 , 0.  , 0.  , 0.  , 0.5 , 1.  ],
             [0.  , 0.25, 0.  , 0.25, 0.  , 0.  ],
             [0.  , 0.  , 0.  , 0.25, 0.  , 0.  ]]
        )

        # When / Then
        np.testing.assert_array_almost_equal(mark(3).todense(), r_mark)

    def test_laplace_5(self):
        ## Given
        r_m = np.array(
            [[-2.,  1.,  0.,  0.,  0.],
             [ 1., -2.,  1.,  0.,  0.],
             [ 0.,  1., -2.,  1.,  0.],
             [ 0.,  0.,  1., -2.,  1.],
             [ 0.,  0.,  0.,  1., -2.]]
        )

        ## When/Then
        m = laplace(5)
        np.testing.assert_array_almost_equal(m.todense(), r_m)

    def test_laplace_eivals(self):
        ## Given
        m = np.array(
            [[-2.,  1.,  0.,  0.,  0.],
             [ 1., -2.,  1.,  0.,  0.],
             [ 0.,  1., -2.,  1.,  0.],
             [ 0.,  0.,  1., -2.,  1.],
             [ 0.,  0.,  0.,  1., -2.]]
        )
        r_eivals = np.sort(nlin.eig(m)[0])[::-1]

        ## When/Then
        np.testing.assert_array_almost_equal(laplace_eigen(5), r_eivals)
