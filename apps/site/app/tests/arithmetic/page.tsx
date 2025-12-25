"use client";

export default function ArithmeticTestPage() {
  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Arithmetic Tests</h1>
      <p style={styles.description}>
        Tests for element-wise arithmetic operations (add, multiply).
      </p>
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Test Cases</h3>
        <ul style={styles.list}>
          <li>Element-wise addition</li>
          <li>Element-wise multiplication</li>
          <li>Broadcasting behavior</li>
          <li>Scalar operations</li>
        </ul>
      </div>
      <div style={styles.notice}>
        <p>Test harness integration pending. See packages/core/tests/components/ArithmeticTests.tsx</p>
      </div>
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  container: { padding: "2rem", maxWidth: "800px" },
  title: { fontSize: "1.75rem", fontWeight: 600, marginBottom: "0.5rem" },
  description: { color: "#666", marginBottom: "1.5rem" },
  card: { background: "#f9f9f9", padding: "1.25rem", borderRadius: "8px", marginBottom: "1rem" },
  cardTitle: { fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem" },
  list: { margin: 0, paddingLeft: "1.25rem", lineHeight: 1.8 },
  notice: { padding: "1rem", background: "#fff3cd", borderRadius: "8px", fontSize: "0.9rem", color: "#856404" },
};
