
/**
 * BLAS Level 1 routines (Vector-Vector operations).
 * Implemented in pure Java using double precision.
 */
public class BLASLevel1 {

    /**
     * DDOT computes the dot product of two vectors.
     * ret = x . y
     *
     * @param n    number of elements
     * @param x    vector x
     * @param offx offset for x
     * @param incx increment for x
     * @param y    vector y
     * @param offy offset for y
     * @param incy increment for y
     * @return dot product
     */
    public static double ddot(int n, double[] x, int offx, int incx, double[] y, int offy, int incy) {
        double dot = 0.0;
        if (n <= 0) return dot;

        int ix = offx;
        int iy = offy;

        for (int i = 0; i < n; i++) {
            dot += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
        return dot;
    }

    /**
     * DAXPY computes y = alpha * x + y
     *
     * @param n    number of elements
     * @param alpha scalar alpha
     * @param x    vector x
     * @param offx offset for x
     * @param incx increment for x
     * @param y    vector y
     * @param offy offset for y
     * @param incy increment for y
     */
    public static void daxpy(int n, double alpha, double[] x, int offx, int incx, double[] y, int offy, int incy) {
        if (n <= 0 || alpha == 0.0) return;

        int ix = offx;
        int iy = offy;

        for (int i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }

    /**
     * DSCAL computes x = alpha * x
     *
     * @param n    number of elements
     * @param alpha scalar alpha
     * @param x    vector x
     * @param offx offset for x
     * @param incx increment for x
     */
    public static void dscal(int n, double alpha, double[] x, int offx, int incx) {
        if (n <= 0 || alpha == 1.0) return;

        int ix = offx;
        for (int i = 0; i < n; i++) {
            x[ix] *= alpha;
            ix += incx;
        }
    }

    /**
     * DNRM2 computes the Euclidean norm of a vector.
     * ret = ||x||_2
     *
     * @param n    number of elements
     * @param x    vector x
     * @param offx offset for x
     * @param incx increment for x
     * @return euclidean norm
     */
    public static double dnrm2(int n, double[] x, int offx, int incx) {
        if (n < 1) return 0.0;

        double scale = 0.0;
        double ssq = 1.0;
        int ix = offx;

        for (int i = 0; i < n; i++) {
            double val = x[ix];
            if (val != 0.0) {
                double absxi = Math.abs(val);
                if (scale < absxi) {
                    ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
                    scale = absxi;
                } else {
                    ssq += (absxi / scale) * (absxi / scale);
                }
            }
            ix += incx;
        }
        return scale * Math.sqrt(ssq);
    }
}
