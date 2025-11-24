package blas;

/**
 * BLAS Level 3 routines (Matrix-Matrix operations).
 * Implemented in pure Java using double precision.
 * Matrices are stored in COLUMN-MAJOR order (Fortran style).
 */
public class BLASLevel3 {

    /**
     * DGEMM performs one of the matrix-matrix operations
     * C := alpha*op(A)*op(B) + beta*C,
     *
     * @param transA if "N", op(A) = A. if "T" or "C", op(A) = A**T.
     * @param transB if "N", op(B) = B. if "T" or "C", op(B) = B**T.
     * @param m      number of rows of C and op(A).
     * @param n      number of columns of C and op(B).
     * @param k      number of columns of op(A) and rows of op(B).
     * @param alpha  scalar alpha.
     * @param a      array containing A.
     * @param offa   offset for A.
     * @param lda    leading dimension of A.
     * @param b      array containing B.
     * @param offb   offset for B.
     * @param ldb    leading dimension of B.
     * @param beta   scalar beta.
     * @param c      array containing C.
     * @param offc   offset for C.
     * @param ldc    leading dimension of C.
     */
    public static void dgemm(String transA, String transB,
            int m, int n, int k,
            double alpha,
            double[] a, int offa, int lda,
            double[] b, int offb, int ldb,
            double beta,
            double[] c, int offc, int ldc) {

        boolean nota = transA.equalsIgnoreCase("N");
        boolean notb = transB.equalsIgnoreCase("N");

        if (m == 0 || n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0))
            return;

        // Scale C by beta
        if (beta != 1.0) {
            if (beta == 0.0) {
                for (int j = 0; j < n; j++) {
                    int idx = offc + j * ldc;
                    for (int i = 0; i < m; i++) {
                        c[idx + i] = 0.0;
                    }
                }
            } else {
                for (int j = 0; j < n; j++) {
                    int idx = offc + j * ldc;
                    for (int i = 0; i < m; i++) {
                        c[idx + i] *= beta;
                    }
                }
            }
        }

        if (alpha == 0.0)
            return;

        if (nota) {
            if (notb) {
                // C := alpha*A*B + beta*C
                // A: m x k, B: k x n
                for (int j = 0; j < n; j++) {
                    if (beta == 0.0) {
                        // C(:,j) initialized above
                    }
                    for (int l = 0; l < k; l++) {
                        double temp = alpha * b[offb + l + j * ldb];
                        if (temp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                c[offc + i + j * ldc] += temp * a[offa + i + l * lda];
                            }
                        }
                    }
                }
            } else {
                // C := alpha*A*B**T + beta*C
                // A: m x k, B: n x k
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        double temp = 0.0;
                        for (int l = 0; l < k; l++) {
                            temp += a[offa + i + l * lda] * b[offb + j + l * ldb];
                        }
                        c[offc + i + j * ldc] += alpha * temp;
                    }
                }
            }
        } else {
            if (notb) {
                // C := alpha*A**T*B + beta*C
                // A: k x m, B: k x n
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        double temp = 0.0;
                        for (int l = 0; l < k; l++) {
                            temp += a[offa + l + i * lda] * b[offb + l + j * ldb];
                        }
                        c[offc + i + j * ldc] += alpha * temp;
                    }
                }
            } else {
                // C := alpha*A**T*B**T + beta*C
                // A: k x m, B: n x k
                for (int j = 0; j < n; j++) {
                    for (int l = 0; l < k; l++) {
                        double temp = alpha * b[offb + j + l * ldb];
                        if (temp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                c[offc + i + j * ldc] += temp * a[offa + l + i * lda];
                            }
                        }
                    }
                }
            }
        }
    }
}
