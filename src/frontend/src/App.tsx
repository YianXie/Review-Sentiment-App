import { useState } from "react";
import "./App.css";

function App() {
    const [review, setReview] = useState("");
    const [sentiment, setSentiment] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    function onSubmit() {
        try {
            setLoading(true);
            fetch(`http://localhost:8000/api/get-review-sentiment/${review}`)
                .then((res) => res.json())
                .then((data) => {
                    console.log(data);
                    setSentiment(data);
                });
        } catch (error) {
            console.error("Error fetching sentiment:", error);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="container">
            <main className="card">
                <h1>Review Sentiment App</h1>
                <p className="subtitle">Enter a review to see its sentiment</p>
                <div className="form">
                    <input
                        className="review-input"
                        type="text"
                        placeholder="Type your review..."
                        onChange={(event) => setReview(event.target.value)}
                        value={review}
                    />
                    <button
                        className="submit-btn"
                        onClick={onSubmit}
                        disabled={loading || !review.trim()}
                    >
                        {loading ? "Loading..." : "Submit"}
                    </button>
                </div>
                {sentiment && (
                    <p className="result">
                        Sentiment: <strong>{sentiment}</strong>
                    </p>
                )}
            </main>
        </div>
    );
}

export default App;
