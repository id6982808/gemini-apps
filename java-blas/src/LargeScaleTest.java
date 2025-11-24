import java.util.Random;

public class LargeScaleTest {

    private static final int SIZE = 1000;
    private static final double EPSILON = 1e-9;
    private static final Random random = new Random(42); // 固定シードで再現性を確保

    public static void main(String[] args) {
        System.out.println("Starting Large Scale Tests (N=" + SIZE + ")...");

        testLevel1_DDOT();
        testLevel2_DGEMV();
        testLevel3_DGEMM();

        System.out.println("All tests finished.");
    }

    private static void testLevel1_DDOT() {
        System.out.println("\n--- Testing Level 1: DDOT ---");
        double[] x = createRandomVector(SIZE);
        double[] y = createRandomVector(SIZE);

        long start = System.nanoTime();
        double result = BLASLevel1.ddot(SIZE, x, 0, 1, y, 0, 1);
        long end = System.nanoTime();

        // Verification (Naive implementation)
        double expected = 0.0;
        for (int i = 0; i < SIZE; i++) {
            expected += x[i] * y[i];
        }

        System.out.printf("BLAS Result: %.6f%n", result);
        System.out.printf("Ref  Result: %.6f%n", expected);
        System.out.printf("Diff       : %.6e%n", Math.abs(result - expected));
        System.out.printf("Time       : %.3f ms%n", (end - start) / 1e6);

        if (Math.abs(result - expected) > EPSILON * SIZE) {
            System.err.println("DDOT FAILED!");
        } else {
            System.out.println("DDOT PASSED");
        }
    }

    private static void testLevel2_DGEMV() {
        System.out.println("\n--- Testing Level 2: DGEMV ---");
        // y = alpha * A * x + beta * y
        int m = SIZE;
        int n = SIZE;
        double alpha = 1.5;
        double beta = 0.5;

        double[] a = createRandomVector(m * n); // Column-major
        double[] x = createRandomVector(n);
        double[] y = createRandomVector(m);
        double[] yRef = y.clone(); // Copy for reference

        long start = System.nanoTime();
        BLASLevel2.dgemv("N", m, n, alpha, a, 0, m, x, 0, 1, beta, y, 0, 1);
        long end = System.nanoTime();

        // Verification
        // yRef = beta * yRef
        for (int i = 0; i < m; i++)
            yRef[i] *= beta;
        // yRef += alpha * A * x
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                // A is column-major: A(i,j) is at a[i + j*m]
                yRef[i] += alpha * a[i + j * m] * x[j];
            }
        }

        double diff = 0.0;
        for (int i = 0; i < m; i++) {
            diff += Math.abs(y[i] - yRef[i]);
        }

        System.out.printf("Total Diff : %.6e%n", diff);
        System.out.printf("Time       : %.3f ms%n", (end - start) / 1e6);

        if (diff > EPSILON * SIZE * SIZE) {
            System.err.println("DGEMV FAILED!");
        } else {
            System.out.println("DGEMV PASSED");
        }
    }

    private static void testLevel3_DGEMM() {
        System.out.println("\n--- Testing Level 3: DGEMM ---");
        // C = alpha * A * B + beta * C
        int m = SIZE;
        int n = SIZE;
        int k = SIZE;
        double alpha = 1.2;
        double beta = 0.8;

        double[] a = createRandomVector(m * k);
        double[] b = createRandomVector(k * n);
        double[] c = createRandomVector(m * n);
        double[] cRef = c.clone();

        System.out.println("Calculating DGEMM (this might take a moment)...");
        long start = System.nanoTime();
        BLASLevel3.dgemm("N", "N", m, n, k, alpha, a, 0, m, b, 0, k, beta, c, 0, m);
        long end = System.nanoTime();

        System.out.printf("Time       : %.3f ms%n", (end - start) / 1e6);

        // Verification (Only check a few random elements to save time,
        // as full O(N^3) verification in Java is slow)
        System.out.println("Verifying random elements...");
        int checks = 10;
        boolean passed = true;
        for (int check = 0; check < checks; check++) {
            int row = random.nextInt(m);
            int col = random.nextInt(n);

            double expected = cRef[row + col * m] * beta;
            // A row i, B col j
            // A(row, l) * B(l, col)
            // A is m x k (col-major): a[row + l*m]
            // B is k x n (col-major): b[l + col*k]
            for (int l = 0; l < k; l++) {
                expected += alpha * a[row + l * m] * b[l + col * k];
            }

            double actual = c[row + col * m];
            if (Math.abs(actual - expected) > 1e-5) { // Slightly larger epsilon for accumulated error
                System.err.printf("Mismatch at (%d, %d): Expected %.6f, Got %.6f%n", row, col, expected, actual);
                passed = false;
                break;
            }
        }

        if (passed) {
            System.out.println("DGEMM PASSED (Random sampling verification)");
        } else {
            System.err.println("DGEMM FAILED!");
        }
    }

    private static double[] createRandomVector(int size) {
        double[] v = new double[size];
        for (int i = 0; i < size; i++) {
            v[i] = random.nextDouble();
        }
        return v;
    }
}
