import React, { useState, useRef } from 'react';

interface HighlightableTextProps {
  text: string;
  onHighlight?: (selectedText: string, context: string) => void;
  onSourceEvidence?: (selectedText: string, context: string) => void;
  className?: string;
}

type HighlightMode = 'source' | 'question';

const HighlightableText: React.FC<HighlightableTextProps> = ({ 
  text, 
  onHighlight, 
  onSourceEvidence, 
  className = ''
}) => {
  const [highlightMode, setHighlightMode] = useState<HighlightMode>('source');
  const textRef = useRef<HTMLDivElement>(null);

  const handleMouseUp = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const selectedString = selection.toString();
      
      // Handle different modes
      if (highlightMode === 'source') {
        handleSourceEvidence(selectedString);
      } else {
        handleAskQuestion(selectedString);
      }
    }
  };

  const handleSourceEvidence = (selectedString: string) => {
    // Get context (surrounding text)
    const textContent = textRef.current?.textContent || '';
    const selectedIndex = textContent.indexOf(selectedString);
    const contextStart = Math.max(0, selectedIndex - 100);
    const contextEnd = Math.min(textContent.length, selectedIndex + selectedString.length + 100);
    const context = textContent.substring(contextStart, contextEnd);
    
    // Call source evidence callback
    if (onSourceEvidence) {
      onSourceEvidence(selectedString, context);
    }
    
    // Clear selection
    window.getSelection()?.removeAllRanges();
  };

  const handleAskQuestion = (selectedString: string) => {
    // Use native prompt for simplicity (as requested)
    const question = prompt("What would you like to know about this text?");
    if (!question) {
      window.getSelection()?.removeAllRanges();
      return;
    }

    // Get context (surrounding text)
    const textContent = textRef.current?.textContent || '';
    const selectedIndex = textContent.indexOf(selectedString);
    const contextStart = Math.max(0, selectedIndex - 100);
    const contextEnd = Math.min(textContent.length, selectedIndex + selectedString.length + 100);
    const context = textContent.substring(contextStart, contextEnd);
    
    // Call highlight callback with question context
    if (onHighlight) {
      onHighlight(selectedString, context);
    }
    
    // Clear selection
    window.getSelection()?.removeAllRanges();
  };

  // Parse text and apply formatting
  const formatText = (text: string) => {
    const lines = text.split('\n');
    return lines.map((line, index) => {
      // Check for bold markers
      const boldPattern = /\*\*(.*?)\*\*/g;
      const parts = [];
      let lastIndex = 0;
      let match;
      
      while ((match = boldPattern.exec(line)) !== null) {
        // Add text before bold
        if (match.index > lastIndex) {
          parts.push(
            <span key={`text-${index}-${lastIndex}`}>
              {line.substring(lastIndex, match.index)}
            </span>
          );
        }
        // Add bold text
        parts.push(
          <strong key={`bold-${index}-${match.index}`} className="font-bold text-blue-800">
            {match[1]}
          </strong>
        );
        lastIndex = match.index + match[0].length;
      }
      
      // Add remaining text
      if (lastIndex < line.length) {
        parts.push(
          <span key={`text-${index}-${lastIndex}-end`}>
            {line.substring(lastIndex)}
          </span>
        );
      }
      
      return (
        <div key={`line-${index}`} className="mb-2">
          {parts.length > 0 ? parts : line}
        </div>
      );
    });
  };

  return (
    <div className="relative">
      {/* Mode Toggle */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setHighlightMode('source')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              highlightMode === 'source'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
            }`}
          >
            Source Evidence
          </button>
          <button
            onClick={() => setHighlightMode('question')}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              highlightMode === 'question'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
            }`}
          >
            Ask About Highlight
          </button>
        </div>
        
        <span className="text-sm text-gray-500">
          {highlightMode === 'source' 
            ? 'Highlight text to see source evidence' 
            : 'Highlight text to ask questions'
          }
        </span>
      </div>

      <div
        ref={textRef}
        onMouseUp={handleMouseUp}
        className={`select-text ${className}`}
        style={{ 
          userSelect: 'text', 
          cursor: 'text'
        }}
      >
        {formatText(text)}
      </div>
    </div>
  );
};

export default HighlightableText; 