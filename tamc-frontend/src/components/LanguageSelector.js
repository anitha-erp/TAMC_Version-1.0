// ===============================================================
// 🌍 LANGUAGE SELECTOR COMPONENT
// Dropdown for switching between Telugu, English, and Hindi
// ===============================================================

import React from 'react';

const LanguageSelector = ({ currentLanguage, onLanguageChange }) => {
    const languages = [
        { code: 'en', name: 'English', flag: '🇬🇧', nativeName: 'English' },
        { code: 'te', name: 'Telugu', flag: 'తె', nativeName: 'తెలుగు' },
        { code: 'hi', name: 'Hindi', flag: 'हिं', nativeName: 'हिंदी' }
    ];

    const currentLang = languages.find(lang => lang.code === currentLanguage) || languages[0];

    return (
        <div style={{
            position: 'relative',
            display: 'inline-block'
        }}>
            <select
                value={currentLanguage}
                onChange={(e) => onLanguageChange(e.target.value)}
                style={{
                    padding: '8px 32px 8px 12px',
                    fontSize: '14px',
                    fontWeight: '600',
                    color: '#374151',
                    background: 'white',
                    border: '2px solid #e5e7eb',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    outline: 'none',
                    transition: 'all 0.2s ease',
                    appearance: 'none',
                    WebkitAppearance: 'none',
                    MozAppearance: 'none',
                    backgroundImage: `url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e")`,
                    backgroundRepeat: 'no-repeat',
                    backgroundPosition: 'right 8px center',
                    backgroundSize: '16px'
                }}
                onMouseEnter={(e) => {
                    e.target.style.borderColor = '#16a34a';
                    e.target.style.boxShadow = '0 0 0 3px rgba(22, 163, 74, 0.1)';
                }}
                onMouseLeave={(e) => {
                    e.target.style.borderColor = '#e5e7eb';
                    e.target.style.boxShadow = 'none';
                }}
            >
                {languages.map(lang => (
                    <option key={lang.code} value={lang.code}>
                        {lang.flag} {lang.nativeName}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default LanguageSelector;
