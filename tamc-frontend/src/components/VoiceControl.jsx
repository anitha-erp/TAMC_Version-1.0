import React from "react";

const VoiceControl = ({ isListening, startListening, language, setLanguage }) => {
  return (
    <div className="voice-control">
      <button
        className={`mic-btn ${isListening ? "active" : ""}`}
        onClick={startListening}
      >
        {isListening ? "🎙️ Listening..." : "🎤 Speak"}
      </button>

      <select
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
        className="lang-select"
      >
        <option value="en-IN">English 🇮🇳</option>
        <option value="hi-IN">Hindi 🇮🇳</option>
        <option value="te-IN">Telugu 🇮🇳</option>
        <option value="ta-IN">Tamil 🇮🇳</option>
      </select>
    </div>
  );
};

export default VoiceControl;
