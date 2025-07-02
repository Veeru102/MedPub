import React from 'react';

export type AudienceType = 'patient' | 'clinician' | 'researcher';

interface AudienceSelectorProps {
  selectedAudience: AudienceType;
  onAudienceChange: (audience: AudienceType) => void;
  disabled?: boolean;
}

const AudienceSelector: React.FC<AudienceSelectorProps> = ({ 
  selectedAudience, 
  onAudienceChange, 
  disabled = false 
}) => {
  const audiences: { value: AudienceType; label: string; description: string; icon: string }[] = [
    {
      value: 'patient',
      label: 'Patient',
      description: 'Simple language, focus on practical implications',
      icon: '👤'
    },
    {
      value: 'clinician',
      label: 'Clinician',
      description: 'Medical terminology, clinical relevance',
      icon: '👨‍⚕️'
    },
    {
      value: 'researcher',
      label: 'Researcher',
      description: 'Technical detail, methodology critique',
      icon: '🔬'
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        Select Summary Style
      </h3>
      <div className="flex space-x-3">
        {audiences.map((audience) => (
          <button
            key={audience.value}
            onClick={() => onAudienceChange(audience.value)}
            disabled={disabled}
            className={`
              flex-1 p-3 rounded-lg border-2 transition-all duration-200
              ${selectedAudience === audience.value
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="text-2xl mb-1">{audience.icon}</div>
            <div className="font-medium text-gray-800">{audience.label}</div>
            <div className="text-xs text-gray-600 mt-1">{audience.description}</div>
          </button>
        ))}
      </div>
      
      {/* Current selection indicator */}
      <div className="mt-3 text-sm text-gray-600">
        Current style: <span className="font-medium text-blue-600">
          {audiences.find(a => a.value === selectedAudience)?.label}
        </span>
      </div>
    </div>
  );
};

export default AudienceSelector; 