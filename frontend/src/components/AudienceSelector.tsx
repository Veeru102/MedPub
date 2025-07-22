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
    <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">Summary Style</h3>
        <span className="text-xs text-gray-600 dark:text-gray-400">
          Current: <span className="font-medium text-blue-600 dark:text-blue-400">{currentAudience?.label}</span>
        </span>
      </div>
      
      <div className="flex space-x-2">
        {audiences.map((audience) => (
          <button
            key={audience.value}
            onClick={() => onAudienceChange(audience.value)}
            disabled={disabled}
            className={`
              flex-1 px-4 py-3 rounded-lg border transition-all duration-200 text-sm
              ${selectedAudience === audience.value
                ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="font-medium">{audience.label}</div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">{audience.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default AudienceSelector; 