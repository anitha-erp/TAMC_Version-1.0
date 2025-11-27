import React from "react";

export default function LanguageSelector({ language, setLanguage }) {
  const languages = [
    { code: "en-IN", label: "English 🇮🇳" },
    { code: "hi-IN", label: "Hindi 🇮🇳" },
    { code: "te-IN", label: "Telugu 🇮🇳" },
    { code: "ta-IN", label: "Tamil 🇮🇳" },
    { code: "kn-IN", label: "Kannada 🇮🇳" },
  ];

  return (
    <select
      value={language}
      onChange={(e) => setLanguage(e.target.value)}
      className="language-select"
    >
      {languages.map((lang) => (
        <option key={lang.code} value={lang.code}>
          {lang.label}
        </option>
      ))}
    </select>
  );
}
