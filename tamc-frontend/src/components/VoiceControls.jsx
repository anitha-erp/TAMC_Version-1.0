import React, { useState } from "react";

export default function VoiceControls({ onResult }) {
  const [listening, setListening] = useState(false);
  let recognition;

  if ("webkitSpeechRecognition" in window) {
    recognition = new window.webkitSpeechRecognition();
    recognition.lang = "en-IN";
    recognition.continuous = false;
    recognition.interimResults = false;
  }

  const startListening = () => {
    if (!recognition) return alert("Speech Recognition not supported in this browser");
    setListening(true);
    recognition.start();

    recognition.onresult = (event) => {
      const spokenText = event.results[0][0].transcript;
      onResult(spokenText);
      setListening(false);
    };

    recognition.onerror = () => setListening(false);
  };

  return (
    <button onClick={startListening} className={`mic-btn ${listening ? "active" : ""}`}>
      🎙️ {listening ? "Listening..." : "Speak"}
    </button>
  );
}
