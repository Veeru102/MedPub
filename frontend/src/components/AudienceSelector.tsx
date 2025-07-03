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
  const audiences: { value: AudienceType; label: string; description: string }[] = [
    {
      value: 'patient',
      label: 'Patient',
      description: 'Simple language, practical focus'
    },
    {
      value: 'clinician',
      label: 'Clinician',
      description: 'Medical terminology, clinical relevance'
    },
    {
      value: 'researcher',
      label: 'Researcher',
      description: 'Technical detail, methodology focus'
    }
  ];

  const currentAudience = audiences.find(a => a.value === selectedAudience);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700">Summary Style</h3>
        <span className="text-xs text-gray-500">
          Current: <span className="font-medium text-blue-600">{currentAudience?.label}</span>
        </span>
      </div>
      
      <div className="flex space-x-2">
        {audiences.map((audience) => (
          <button
            key={audience.value}
            onClick={() => onAudienceChange(audience.value)}
            disabled={disabled}
            className={`
              flex-1 px-3 py-2 rounded-md border transition-all duration-200 text-sm
              ${selectedAudience === audience.value
                ? 'border-blue-500 bg-blue-50 text-blue-700'
                : 'border-gray-200 text-gray-700 hover:border-gray-300 hover:bg-gray-50'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="font-medium">{audience.label}</div>
            <div className="text-xs text-gray-500 mt-0.5">{audience.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default AudienceSelector; 