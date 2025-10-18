import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [source, setSource] = useState("");
  const [requiresFeedback, setRequiresFeedback] = useState(false);
  const [feedback, setFeedback] = useState("");

  const handleSubmit = async () => {
    if (!question.trim()) return;
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/solve?question=${encodeURIComponent(question)}`
      );
      const data = await res.json();
      setAnswer(data.answer);
      setSource(data.source);
      setRequiresFeedback(data.source === "Web");
    } catch (err) {
      setAnswer("Error fetching answer.");
      setSource("");
      setRequiresFeedback(false);
    }
  };

  const handleFeedbackSubmit = async () => {
    if (!feedback.trim()) return;
    try {
      await fetch("http://127.0.0.1:8000/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, corrected_answer: feedback }),
      });
      setAnswer(feedback);
      setSource("KB (updated by user)");
      setRequiresFeedback(false);
      setFeedback("");
    } catch (err) {
      console.error("Error submitting feedback:", err);
    }
  };

  const styles = {
    header: {
      padding: "20px 40px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      background: "linear-gradient(90deg, #4f46e5, #8b5cf6)",
      color: "white",
      position: "sticky",
      top: 0,
      zIndex: 100,
      borderBottom: "2px solid #e0e7ff",
      transition: "all 0.3s ease-in-out",
    },
    navLink: { margin: "0 15px", color: "white", cursor: "pointer", fontWeight: 500, transition: "0.3s" },
    section: { padding: "80px 20px", textAlign: "center" },
    hero: { background: "linear-gradient(160deg, #e0f2fe, #dbeafe)", padding: "120px 20px", position: "relative", overflow: "hidden" },
    featureCard: { background: "white", padding: "25px", borderRadius: "16px", width: "250px", boxShadow: "0 12px 25px rgba(0,0,0,0.1)", transition: "transform 0.3s, box-shadow 0.3s", cursor: "pointer" },
    mathCard: { background: "#f8fafc", padding: "30px", borderRadius: "16px", maxWidth: "600px", margin: "40px auto", boxShadow: "0 12px 25px rgba(0,0,0,0.1)" },
    input: { padding: "10px", width: "70%", borderRadius: "8px", marginRight: "10px", border: "1px solid #d1d5db" },
    button: { padding: "10px 15px", borderRadius: "8px", background: "#4f46e5", color: "white", border: "none", cursor: "pointer", transition: "0.3s" },
    feedbackInput: { padding: "8px", width: "60%", borderRadius: "8px", marginRight: "10px", border: "1px solid #d1d5db" },
    footer: { textAlign: "center", padding: "30px 20px", background: "#4f46e5", color: "white" },
    animatedTitle: { display: "inline-block", animation: "bounce 1.5s infinite" },
    floatingSymbol: { position: "absolute", fontSize: "1.5rem", color: "#4f46e5", opacity: 0.3 }
  };

  // Floating symbols positions
  const symbols = ["π", "Σ", "√", "∞", "Δ", "∫", "AI"];
  const symbolElements = symbols.map((sym, i) => {
    const left = Math.random() * 100;
    const top = Math.random() * 100;
    const duration = 10 + Math.random() * 10;
    return (
      <span
        key={i}
        style={{
          ...styles.floatingSymbol,
          left: `${left}%`,
          top: `${top}%`,
          animation: `float ${duration}s ease-in-out infinite alternate`,
        }}
      >
        {sym}
      </span>
    );
  });

  return (
    <div>
      {/* Navbar */}
      <header style={styles.header}>
        <h2 style={{ fontWeight: 700, fontSize: "1.8rem" }}>Math Professor Agent 🤖</h2>
        <nav>
          <span style={styles.navLink} onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>Home</span>
          <span style={styles.navLink} onClick={() => document.getElementById("features").scrollIntoView({ behavior: "smooth" })}>Features</span>
          <span style={styles.navLink} onClick={() => document.getElementById("agent").scrollIntoView({ behavior: "smooth" })}>Try it Now</span>
          <span style={styles.navLink} onClick={() => document.getElementById("about").scrollIntoView({ behavior: "smooth" })}>About</span>
        </nav>
      </header>

      {/* Hero Section */}
      <section id="hero" style={styles.hero}>
        {symbolElements}
        <h1 style={{ fontSize: "3rem", color: "#4f46e5" }}>
          Learn Math with Your <span style={styles.animatedTitle}>AI Professor</span>
        </h1>
        <p style={{ fontSize: "1.3rem", color: "#374151", maxWidth: "600px", margin: "20px auto" }}>
          Step-by-step guidance, interactive solutions, and personalized feedback to enhance your learning.
        </p>
      </section>

      {/* Features Section */}
      <section id="features" style={{ ...styles.section, background: "#f8fafc" }}>
        <h2 style={{ fontSize: "2.5rem", color: "#4f46e5" }}>How Math Professor Agent Helps You</h2>
        <div style={{ display: "flex", justifyContent: "center", flexWrap: "wrap", marginTop: "40px", gap: "20px" }}>
          {[
            { title: "Step-by-Step Guidance", desc: "Understand every step in your math problems." },
            { title: "Feedback Learning", desc: "Correct and improve answers with human feedback." },
            { title: "Reliable Sources", desc: "Web search fallback ensures accurate information." },
            { title: "Interactive Learning", desc: "Engage with questions, try multiple problems, and learn faster." }
          ].map((feature, i) => (
            <div key={i} style={styles.featureCard} className="featureCardHover">
              <h3 style={{ color: "#4f46e5" }}>{feature.title}</h3>
              <p style={{ color: "#374151", marginTop: "10px" }}>{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Math Agent Section */}
      <section id="agent" style={styles.mathCard}>
        <h2 style={{ color: "#4f46e5" }}>Ask a Math Question</h2>
        <input style={styles.input} type="text" placeholder="Enter math question..." value={question} onChange={(e) => setQuestion(e.target.value)} />
        <button style={styles.button} onClick={handleSubmit}>Solve</button>

        {answer && (
          <div style={{ marginTop: "20px", textAlign: "left", animation: "fadeInUp 0.6s ease-in" }}>
            <p><strong>Answer:</strong> {answer}</p>
            <p><em>Source: {source}</em></p>

            {requiresFeedback && (
              <div style={{ marginTop: "10px" }}>
                <input style={styles.feedbackInput} type="text" placeholder="Correct the answer if needed" value={feedback} onChange={(e) => setFeedback(e.target.value)} />
                <button style={styles.button} onClick={handleFeedbackSubmit}>Submit Feedback</button>
              </div>
            )}
          </div>
        )}
      </section>

      {/* About Section */}
      <section id="about" style={{ ...styles.section, background: "#f0f4ff" }}>
        <h2 style={{ color: "#4f46e5", fontSize: "2.5rem" }}>About the AI Math Professor</h2>
        <p style={{ maxWidth: "700px", margin: "20px auto", color: "#374151", fontSize: "1.2rem" }}>
          This Math Agent acts like your personal professor. It helps students understand math concepts,
          guides them step-by-step, allows corrections through feedback, and ensures learning is interactive and personalized.
          It’s like having a virtual tutor available 24/7.
        </p>
      </section>

      {/* Footer */}
      <footer style={styles.footer}>
        &copy; {new Date().getFullYear()} Math Professor AI Agent. All rights reserved.
      </footer>

      {/* Animations & floating symbols */}
      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }
        @keyframes float {
          0% { transform: translateY(0px) translateX(0px); }
          50% { transform: translateY(-20px) translateX(10px); }
          100% { transform: translateY(0px) translateX(0px); }
        }
        .featureCardHover:hover {
          transform: translateY(-10px);
          box-shadow: 0 20px 30px rgba(0,0,0,0.15);
        }
        button:hover {
          transform: scale(1.05);
          background: #6366f1;
        }
        input:focus {
          outline: none;
          border-color: #4f46e5;
          box-shadow: 0 0 8px rgba(79,70,229,0.3);
        }
      `}</style>
    </div>
  );
}

export default App;



