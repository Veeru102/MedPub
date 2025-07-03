import React, { useState, useRef } from 'react';
import QuestionModal from './QuestionModal';

interface HighlightableTextProps {
  text: string;
  onHighlight?: (selectedText: string, context: string, question: string) => void;
  onSourceEvidence?: (selectedText: string, context: string) => void;
  className?: string;
}

type HighlightMode = 'source' | 'question';

interface HighlightedRange {
  text: string;
  id: string;
}

const HighlightableText: React.FC<HighlightableTextProps> = ({ 
  text, 
  onHighlight, 
  onSourceEvidence, 
  className = ''
}) => {
  const [highlightMode, setHighlightMode] = useState<HighlightMode>('source');
  const [showQuestionModal, setShowQuestionModal] = useState(false);
  const [currentSelectedText, setCurrentSelectedText] = useState('');
  const [currentContext, setCurrentContext] = useState('');
  const [highlightedRanges, setHighlightedRanges] = useState<HighlightedRange[]>([]);
  const textRef = useRef<HTMLDivElement>(null);

  const handleMouseUp = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const selectedString = selection.toString();
      
      // Get context (surrounding text)
      const textContent = textRef.current?.textContent || '';
      const selectedIndex = textContent.indexOf(selectedString);
      const contextStart = Math.max(0, selectedIndex - 100);
      const contextEnd = Math.min(textContent.length, selectedIndex + selectedString.length + 100);
      const context = textContent.substring(contextStart, contextEnd);
      
      // Add persistent highlight (simplified approach)
      const newHighlight: HighlightedRange = {
        text: selectedString,
        id: Date.now().toString()
      };
      
      // Avoid duplicate highlights
      const isDuplicate = highlightedRanges.some(highlight => highlight.text === selectedString);
      if (!isDuplicate) {
        setHighlightedRanges(prev => [...prev, newHighlight]);
      }
      
      // Handle different modes
      if (highlightMode === 'source') {
        handleSourceEvidence(selectedString, context);
      } else {
        // Store selection and context for modal
        setCurrentSelectedText(selectedString);
        setCurrentContext(context);
        setShowQuestionModal(true);
      }
      
      // Clear browser selection but keep our highlight
      window.getSelection()?.removeAllRanges();
    }
  };

  const handleSourceEvidence = (selectedString: string, context: string) => {
    // Call source evidence callback
    if (onSourceEvidence) {
      onSourceEvidence(selectedString, context);
    }
  };

  const handleQuestionSubmit = (question: string) => {
    // Call highlight callback with question context
    if (onHighlight) {
      onHighlight(currentSelectedText, currentContext, question);
    }
  };

  const handleModalClose = () => {
    setShowQuestionModal(false);
    setCurrentSelectedText('');
    setCurrentContext('');
  };

  // Enhanced text formatting function that properly handles markdown and highlights
  const formatText = (text: string) => {
    if (!text) return '';

    const lines = text.split('\n').filter(line => line.trim() !== '');
    
    return lines.map((line, lineIndex) => {
      const trimmedLine = line.trim();
      
      // Handle headers (### ## #)
      if (trimmedLine.match(/^#{1,3}\s/)) {
        const level = trimmedLine.match(/^(#{1,3})/)?.[1].length || 3;
        const content = trimmedLine.replace(/^#{1,3}\s/, '').trim();
        
        if (level === 1) {
          return <h3 key={`header-${lineIndex}`} className="text-lg font-bold text-blue-800 mb-3 mt-4">{renderTextWithHighlights(content)}</h3>;
        } else if (level === 2) {
          return <h4 key={`header-${lineIndex}`} className="text-base font-semibold text-blue-700 mb-2 mt-3">{renderTextWithHighlights(content)}</h4>;
        } else {
          return <h5 key={`header-${lineIndex}`} className="text-sm font-medium text-blue-600 mb-2 mt-3">{renderTextWithHighlights(content)}</h5>;
        }
      }

      // Handle bullet points (- or * at start)
      if (trimmedLine.match(/^\s*[\-\*]\s/)) {
        const content = trimmedLine.replace(/^\s*[\-\*]\s/, '').trim();
        return (
          <div key={`bullet-${lineIndex}`} className="flex items-start mb-2 ml-4">
            <span className="text-blue-600 mr-3 mt-1 font-bold">•</span>
            <span className="flex-1">{renderTextWithHighlights(formatInlineText(content))}</span>
          </div>
        );
      }

      // Handle numbered lists (1. 2. etc.)
      if (trimmedLine.match(/^\s*\d+\.\s/)) {
        const number = trimmedLine.match(/^\s*(\d+)\./)?.[1] || '1';
        const content = trimmedLine.replace(/^\s*\d+\.\s/, '').trim();
        return (
          <div key={`numbered-${lineIndex}`} className="flex items-start mb-2 ml-4">
            <span className="text-blue-600 mr-3 mt-1 font-semibold min-w-[1.5rem]">{number}.</span>
            <span className="flex-1">{renderTextWithHighlights(formatInlineText(content))}</span>
          </div>
        );
      }

      // Regular paragraph
      return (
        <p key={`para-${lineIndex}`} className="mb-3 leading-relaxed">
          {renderTextWithHighlights(formatInlineText(trimmedLine))}
        </p>
      );
    });
  };

  // Render text with persistent highlights (simplified approach)
  const renderTextWithHighlights = (content: React.ReactNode): React.ReactNode => {
    if (typeof content !== 'string') {
      return content;
    }

    if (highlightedRanges.length === 0) {
      return content;
    }

    let result: React.ReactNode = content;
    
    // Process each highlight
    highlightedRanges.forEach((highlight) => {
      if (typeof result === 'string' && result.includes(highlight.text)) {
        const parts = result.split(highlight.text);
        const highlightedParts: React.ReactNode[] = [];
        
        parts.forEach((part, index) => {
          highlightedParts.push(part);
          if (index < parts.length - 1) {
            highlightedParts.push(
              <span 
                key={`highlight-${highlight.id}-${index}`}
                className="px-1 rounded border"
                style={{ 
                  backgroundColor: '#e0f2ff', 
                  borderColor: '#b3d9ff',
                  color: '#1e40af'
                }}
                title="Previously selected text"
              >
                {highlight.text}
              </span>
            );
          }
        });
        
        result = highlightedParts;
      }
    });
    
    return result;
  };

  // Format inline text elements (bold, italic, etc.)
  const formatInlineText = (text: string): React.ReactNode => {
    const parts = [];
    let currentIndex = 0;
    
    // Handle bold text (**text** or __text__)
    const boldRegex = /(\*\*|__)(.*?)\1/g;
    let match;
    
    while ((match = boldRegex.exec(text)) !== null) {
      // Add text before bold
      if (match.index > currentIndex) {
        parts.push(text.substring(currentIndex, match.index));
      }
      
      // Add bold text
      parts.push(
        <strong key={`bold-${match.index}`} className="font-semibold text-blue-800">
          {match[2]}
        </strong>
      );
      
      currentIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (currentIndex < text.length) {
      parts.push(text.substring(currentIndex));
    }
    
    return parts.length > 0 ? parts : text;
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
            Ask AI Questions
          </button>
        </div>
        
        <div className="flex items-center space-x-3">
          {highlightedRanges.length > 0 && (
            <button
              onClick={() => setHighlightedRanges([])}
              className="text-xs text-red-600 hover:text-red-800 px-2 py-1 rounded border border-red-300 hover:bg-red-50 transition-colors"
              title="Clear all highlights"
            >
              Clear Highlights ({highlightedRanges.length})
            </button>
          )}
          <span className="text-sm text-gray-500">
            {highlightMode === 'source' 
              ? 'Highlight text to see source evidence' 
              : 'Highlight text to ask AI questions'
            }
          </span>
        </div>
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

      {/* Question Modal */}
      <QuestionModal
        isOpen={showQuestionModal}
        selectedText={currentSelectedText}
        onClose={handleModalClose}
        onSubmit={handleQuestionSubmit}
      />
    </div>
  );
};

export default HighlightableText; 