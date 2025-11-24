
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        testLevel1();
        testLevel2();
        testLevel3();
    }

    private static void testLevel1() {
        System.out.println("Testing BLAS Level 1 (DDOT)...");
        int n = 3;
        double[] x = { 1.0, 2.0, 3.0 };
        double[] y = { 4.0, 5.0, 6.0 };
        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        double dot = BLASLevel1.ddot(n, x, 0, 1, y, 0, 1);
        System.out.println("x . y = " + dot + " (Expected: 32.0)");
        System.out.println();
    }

    private static void testLevel2() {
        System.out.println("Testing BLAS Level 2 (DGEMV)...");
        // y = alpha * A * x + beta * y
        // A = [[1, 2], [3, 4]] (2x2)
        // x = [1, 1]
        // y = [0, 0]
        // alpha = 1, beta = 0
        // A*x = [1*1 + 2*1, 3*1 + 4*1] = [3, 7]

        // Column-major storage for A:
        // A(0,0)=1, A(1,0)=3, A(0,1)=2, A(1,1)=4
        double[] a = { 1.0, 3.0, 2.0, 4.0 };
        double[] x = { 1.0, 1.0 };
        double[] y = { 0.0, 0.0 };

        int m = 2;
        int n = 2;
        int lda = 2;

        BLASLevel2.dgemv("N", m, n, 1.0, a, 0, lda, x, 0, 1, 0.0, y, 0, 1);

        System.out.println("y = " + Arrays.toString(y) + " (Expected: [3.0, 7.0])");
        System.out.println();
    }

    private static void testLevel3() {
        System.out.println("Testing BLAS Level 3 (DGEMM)...");
        // C = alpha * A * B + beta * C
        // A = [[1, 2], [3, 4]]
        // B = [[1, 0], [0, 1]] (Identity)
        // C = [[0, 0], [0, 0]]
        // Result should be A

        // Column-major A: 1, 3, 2, 4
        double[] a = { 1.0, 3.0, 2.0, 4.0 };
        // Column-major B: 1, 0, 0, 1
        double[] b = { 1.0, 0.0, 0.0, 1.0 };
        double[] c = { 0.0, 0.0, 0.0, 0.0 };

        int m = 2;
        int n = 2;
        int k = 2;
        int lda = 2;
        int ldb = 2;
        int ldc = 2;

        BLASLevel3.dgemm("N", "N", m, n, k, 1.0, a, 0, lda, b, 0, ldb, 0.0, c, 0, ldc);

        // C is column-major: c[0]=C(0,0), c[1]=C(1,0), c[2]=C(0,1), c[3]=C(1,1)
        // Expected: 1, 3, 2, 4
        System.out.println("C (col-major) = " + Arrays.toString(c) + " (Expected: [1.0, 3.0, 2.0, 4.0])");
    }
}
