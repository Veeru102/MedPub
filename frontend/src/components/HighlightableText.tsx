import React, { useState, useRef, useEffect } from 'react';

interface HighlightableTextProps {
  text: string;
  onHighlight: (selectedText: string, context: string) => void;
  className?: string;
}



const HighlightableText: React.FC<HighlightableTextProps> = ({ text, onHighlight, className = '' }) => {
  const [selectedText, setSelectedText] = useState('');
  const [showExplanationDialog, setShowExplanationDialog] = useState(false);
  const [question, setQuestion] = useState('');
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const textRef = useRef<HTMLDivElement>(null);

  const handleMouseUp = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const selectedString = selection.toString();
      setSelectedText(selectedString);
      
      // Get position for popup
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      setPosition({
        x: rect.left + rect.width / 2,
        y: rect.top - 10
      });
      
      setShowExplanationDialog(true);
    }
  };

  const handleAskQuestion = () => {
    if (selectedText && question) {
      // Get context (surrounding text)
      const textContent = textRef.current?.textContent || '';
      const selectedIndex = textContent.indexOf(selectedText);
      const contextStart = Math.max(0, selectedIndex - 100);
      const contextEnd = Math.min(textContent.length, selectedIndex + selectedText.length + 100);
      const context = textContent.substring(contextStart, contextEnd);
      
      onHighlight(selectedText, context);
      setShowExplanationDialog(false);
      setQuestion('');
      
      // Clear selection
      window.getSelection()?.removeAllRanges();
    }
  };

  const handleCancel = () => {
    setShowExplanationDialog(false);
    setQuestion('');
    window.getSelection()?.removeAllRanges();
  };

  // Close dialog when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (showExplanationDialog && target && !target.closest('.explanation-dialog')) {
        handleCancel();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showExplanationDialog]);

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

      {/* Explanation Dialog */}
      {showExplanationDialog && (
        <div
          className="explanation-dialog fixed z-50 bg-white rounded-lg shadow-xl border border-gray-200 p-4 max-w-md"
          style={{
            left: `${position.x}px`,
            top: `${position.y}px`,
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="mb-3">
            <p className="text-sm text-gray-600 mb-2">
              Selected: <span className="font-medium">"{selectedText.substring(0, 50)}..."</span>
            </p>
            <input
              type="text"
              placeholder="Ask a question about this text..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              autoFocus
            />
          </div>
          <div className="flex justify-end space-x-2">
            <button
              onClick={handleCancel}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleAskQuestion}
              disabled={!question}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              Ask
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default HighlightableText; 