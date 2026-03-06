import { useEffect } from "react";
import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  useEffect(() => {
    // Redirect to the backend which serves the HTML frontend
    window.location.href = BACKEND_URL;
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>Redirecting to Dry Bean Classification System...</p>
      </header>
    </div>
  );
}

export default App;
