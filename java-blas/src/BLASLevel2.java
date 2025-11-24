package blas;

/**
 * BLAS Level 2 routines (Matrix-Vector operations).
 * Implemented in pure Java using double precision.
 * Matrices are stored in COLUMN-MAJOR order (Fortran style).
 * A(i, j) is located at a[offset + i + j * lda].
 */
public class BLASLevel2 {

    /**
     * DGEMV performs one of the matrix-vector operations
     * y := alpha*A*x + beta*y, or y := alpha*A**T*x + beta*y,
     *
     * @param trans if "N", op(A) = A. if "T" or "C", op(A) = A**T.
     * @param m     number of rows of the matrix A.
     * @param n     number of columns of the matrix A.
     * @param alpha scalar alpha.
     * @param a     array containing the matrix A.
     * @param offa  offset for A.
     * @param lda   leading dimension of A.
     * @param x     vector x.
     * @param offx  offset for x.
     * @param incx  increment for x.
     * @param beta  scalar beta.
     * @param y     vector y.
     * @param offy  offset for y.
     * @param incy  increment for y.
     */
    public static void dgemv(String trans, int m, int n, double alpha,
            double[] a, int offa, int lda,
            double[] x, int offx, int incx,
            double beta,
            double[] y, int offy, int incy) {

        boolean transpose = trans.equalsIgnoreCase("T") || trans.equalsIgnoreCase("C");

        // Quick return if possible.
        if ((m == 0) || (n == 0) || ((alpha == 0.0) && (beta == 1.0)))
            return;

        int lenX = transpose ? m : n;
        int lenY = transpose ? n : m;

        // First form y := beta*y.
        if (beta != 1.0) {
            if (incy == 1) {
                if (beta == 0.0) {
                    for (int i = 0; i < lenY; i++)
                        y[offy + i] = 0.0;
                } else {
                    for (int i = 0; i < lenY; i++)
                        y[offy + i] *= beta;
                }
            } else {
                int iy = offy;
                if (beta == 0.0) {
                    for (int i = 0; i < lenY; i++) {
                        y[iy] = 0.0;
                        iy += incy;
                    }
                } else {
                    for (int i = 0; i < lenY; i++) {
                        y[iy] *= beta;
                        iy += incy;
                    }
                }
            }
        }

        if (alpha == 0.0)
            return;

        if (!transpose) {
            // Form y := alpha*A*x + y.
            // A is m x n. x is n. y is m.
            // Column-major: A(i, j) = a[i + j*lda]

            int jx = offx;
            for (int j = 0; j < n; j++) {
                double temp = alpha * x[jx];
                if (temp != 0.0) {
                    int iy = offy;
                    int ia = offa + j * lda;
                    for (int i = 0; i < m; i++) {
                        y[iy] += temp * a[ia + i];
                        iy += incy;
                    }
                }
                jx += incx;
            }
        } else {
            // Form y := alpha*A**T*x + y.
            // A is m x n. A**T is n x m. x is m. y is n.

            int jy = offy;
            for (int j = 0; j < n; j++) {
                double temp = 0.0;
                int ix = offx;
                int ia = offa + j * lda;
                for (int i = 0; i < m; i++) {
                    temp += a[ia + i] * x[ix];
                    ix += incx;
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        }
    }
}
