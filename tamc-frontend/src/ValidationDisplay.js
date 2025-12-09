// ValidationDisplay.js - AI Validation Components
// Add this to your App.js file

import React, { useState } from 'react';

/**
 * Validation Badge Component
 * Displays confidence level with color coding
 */
export const ValidationBadge = ({ validation }) => {
    if (!validation || !validation.confidence) return null;

    const { confidence_score, confidence_level } = validation.confidence;

    const badgeStyles = {
        high: {
            bg: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            border: '#059669',
            icon: '✓'
        },
        medium: {
            bg: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
            border: '#d97706',
            icon: '⚡'
        },
        low: {
            bg: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
            border: '#dc2626',
            icon: '⚠'
        }
    };

    const style = badgeStyles[confidence_level] || badgeStyles.medium;

    return (
        <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            padding: '8px 16px',
            borderRadius: '20px',
            background: style.bg,
            border: `2px solid ${style.border}`,
            color: 'white',
            fontWeight: '600',
            fontSize: '0.9rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            marginTop: '12px',
            marginBottom: '8px'
        }}>
            <span style={{ marginRight: '8px', fontSize: '1.1rem' }}>{style.icon}</span>
            <span>{confidence_level.toUpperCase()} Confidence ({confidence_score}%)</span>
        </div>
    );
};

/**
 * Anomaly Warning Component
 * Displays warnings when anomalies are detected
 */
export const AnomalyWarning = ({ anomalies }) => {
    if (!anomalies || !anomalies.has_anomaly) return null;

    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'high': return { bg: '#fee2e2', border: '#dc2626', text: '#991b1b' };
            case 'medium': return { bg: '#fed7aa', border: '#ea580c', text: '#9a3412' };
            default: return { bg: '#fef3c7', border: '#f59e0b', text: '#92400e' };
        }
    };

    const colors = getSeverityColor(anomalies.severity);

    return (
        <div style={{
            marginTop: '12px',
            padding: '12px 16px',
            background: colors.bg,
            borderLeft: `4px solid ${colors.border}`,
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
            <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                <span style={{
                    color: colors.border,
                    marginRight: '12px',
                    fontSize: '1.3rem'
                }}>⚠️</span>
                <div style={{ flex: 1 }}>
                    <p style={{
                        fontWeight: '700',
                        color: colors.text,
                        margin: '0 0 6px 0',
                        fontSize: '0.95rem'
                    }}>
                        Anomaly Detected
                    </p>
                    <p style={{
                        color: colors.text,
                        margin: '0 0 8px 0',
                        fontSize: '0.85rem'
                    }}>
                        {anomalies.summary}
                    </p>
                    {anomalies.anomalies && anomalies.anomalies.length > 0 && (
                        <div style={{ marginTop: '8px' }}>
                            {anomalies.anomalies.map((anomaly, idx) => (
                                <p key={idx} style={{
                                    fontSize: '0.8rem',
                                    color: colors.text,
                                    margin: '4px 0',
                                    paddingLeft: '12px',
                                    borderLeft: `2px solid ${colors.border}`,
                                    opacity: 0.9
                                }}>
                                    • {anomaly.message}
                                </p>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

/**
 * Validation Details Component
 * Expandable section showing detailed validation info
 */
export const ValidationDetails = ({ validation }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!validation || !validation.confidence) return null;

    const { confidence, anomalies } = validation;

    return (
        <details
            open={isExpanded}
            onToggle={(e) => setIsExpanded(e.target.open)}
            style={{ marginTop: '12px' }}
        >
            <summary style={{
                cursor: 'pointer',
                fontSize: '0.85rem',
                color: '#6b7280',
                fontWeight: '600',
                padding: '8px 12px',
                background: '#f9fafb',
                borderRadius: '6px',
                border: '1px solid #e5e7eb',
                userSelect: 'none'
            }}>
                📊 View Validation Details
            </summary>
            <div style={{
                marginTop: '8px',
                padding: '12px',
                background: '#f9fafb',
                borderRadius: '6px',
                fontSize: '0.85rem',
                border: '1px solid #e5e7eb'
            }}>
                <div style={{ marginBottom: '12px' }}>
                    <strong style={{ color: '#374151' }}>AI Reasoning:</strong>
                    <p style={{ margin: '4px 0 0 0', color: '#6b7280' }}>
                        {confidence.reasoning}
                    </p>
                </div>

                <div>
                    <strong style={{ color: '#374151', display: 'block', marginBottom: '6px' }}>
                        Confidence Factors:
                    </strong>
                    <ul style={{
                        listStyle: 'none',
                        padding: 0,
                        margin: 0,
                        display: 'grid',
                        gap: '6px'
                    }}>
                        <li style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            padding: '6px 8px',
                            background: 'white',
                            borderRadius: '4px',
                            border: '1px solid #e5e7eb'
                        }}>
                            <span style={{ color: '#6b7280' }}>Data Quality:</span>
                            <span style={{ fontWeight: '600', color: '#374151' }}>
                                {confidence.factors.data_quality}%
                            </span>
                        </li>
                        <li style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            padding: '6px 8px',
                            background: 'white',
                            borderRadius: '4px',
                            border: '1px solid #e5e7eb'
                        }}>
                            <span style={{ color: '#6b7280' }}>Weather Impact:</span>
                            <span style={{ fontWeight: '600', color: '#374151' }}>
                                {confidence.factors.weather_impact}%
                            </span>
                        </li>
                        <li style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            padding: '6px 8px',
                            background: 'white',
                            borderRadius: '4px',
                            border: '1px solid #e5e7eb'
                        }}>
                            <span style={{ color: '#6b7280' }}>Volatility:</span>
                            <span style={{ fontWeight: '600', color: '#374151' }}>
                                {confidence.factors.volatility}%
                            </span>
                        </li>
                    </ul>
                </div>

                {anomalies && !anomalies.has_anomaly && (
                    <div style={{
                        marginTop: '12px',
                        padding: '8px 12px',
                        background: '#d1fae5',
                        borderRadius: '4px',
                        color: '#065f46',
                        fontSize: '0.8rem',
                        fontWeight: '600'
                    }}>
                        ✓ No anomalies detected - predictions look normal
                    </div>
                )}
            </div>
        </details>
    );
};

/**
 * Complete Validation Display Component
 * Combines all validation components
 */
export const ValidationDisplay = ({ validation }) => {
    if (!validation) return null;

    return (
        <div style={{ marginTop: '16px', marginBottom: '16px' }}>
            <ValidationBadge validation={validation} />
            <AnomalyWarning anomalies={validation.anomalies} />
            <ValidationDetails validation={validation} />
        </div>
    );
};
