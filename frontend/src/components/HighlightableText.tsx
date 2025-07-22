import React, { useState, useRef, useCallback, useMemo } from 'react';
import QuestionModal from './QuestionModal';

interface HighlightableTextProps {
  text: string;
  onHighlight?: (selectedText: string, context: string, question: string) => void;
  onSourceEvidence?: (selectedText: string, context: string) => void;
  className?: string;
}

type HighlightMode = 'source' | 'question';

interface TextHighlight {
  id: string;
  text: string;
  startIndex: number;
  endIndex: number;
  isNew?: boolean;
}

interface TextSegment {
  text: string;
  isHighlighted: boolean;
  highlightId?: string;
  isNew?: boolean;
}

// hook for managing text highlights
const useHighlightText = (originalText: string) => {
  const [highlights, setHighlights] = useState<TextHighlight[]>([]);

  const addHighlight = useCallback((text: string, startIndex: number, endIndex: number) => {
    const newHighlight: TextHighlight = {
      id: Date.now().toString(),
      text,
      startIndex,
      endIndex,
      isNew: true
    };

    // Check for overlapping highlights
    const hasOverlap = highlights.some(h => 
      (startIndex >= h.startIndex && startIndex < h.endIndex) ||
      (endIndex > h.startIndex && endIndex <= h.endIndex) ||
      (startIndex <= h.startIndex && endIndex >= h.endIndex)
    );

    if (!hasOverlap) {
      setHighlights(prev => [...prev, newHighlight]);
      
      // Remove isNew flag after animation
      setTimeout(() => {
        setHighlights(prev => 
          prev.map(h => h.id === newHighlight.id ? { ...h, isNew: false } : h)
        );
      }, 1000);
      
      return true;
    }
    return false;
  }, [highlights]);

  const clearHighlights = useCallback(() => {
    setHighlights([]);
  }, []);

  // Convert text with highlights into segments for rendering
  const textSegments = useMemo((): TextSegment[] => {
    if (!originalText || highlights.length === 0) {
      return [{ text: originalText, isHighlighted: false }];
    }

    // Sort highlights by start index
    const sortedHighlights = [...highlights].sort((a, b) => a.startIndex - b.startIndex);
    const segments: TextSegment[] = [];
    let currentIndex = 0;

    sortedHighlights.forEach(highlight => {
      // Add text before highlight
      if (currentIndex < highlight.startIndex) {
        segments.push({
          text: originalText.slice(currentIndex, highlight.startIndex),
          isHighlighted: false
        });
      }

      // Add highlighted text
      segments.push({
        text: highlight.text,
        isHighlighted: true,
        highlightId: highlight.id,
        isNew: highlight.isNew
      });

      currentIndex = highlight.endIndex;
    });

    // Add remaining text
    if (currentIndex < originalText.length) {
      segments.push({
        text: originalText.slice(currentIndex),
        isHighlighted: false
      });
    }

    return segments;
  }, [originalText, highlights]);

  return {
    highlights,
    textSegments,
    addHighlight,
    clearHighlights,
    highlightCount: highlights.length
  };
};

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
  const textRef = useRef<HTMLDivElement>(null);
  
  const { textSegments, addHighlight, clearHighlights, highlightCount } = useHighlightText(text);

  const handleMouseUp = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || !textRef.current) return;
    
    const selectedText = selection.toString().trim();
    if (!selectedText) return;

    // Get the full text content for context and positioning
    const fullText = textRef.current.textContent || '';
    const range = selection.getRangeAt(0);
    
    // Find start and end positions in the full text
    const preSelectionRange = document.createRange();
    preSelectionRange.selectNodeContents(textRef.current);
    preSelectionRange.setEnd(range.startContainer, range.startOffset);
    const startIndex = preSelectionRange.toString().length;
    const endIndex = startIndex + selectedText.length;

    // Add highlight to state
    const highlightAdded = addHighlight(selectedText, startIndex, endIndex);
    
    if (highlightAdded) {
      // Get context (surrounding text)
      const contextStart = Math.max(0, startIndex - 100);
      const contextEnd = Math.min(fullText.length, endIndex + 100);
      const context = fullText.substring(contextStart, contextEnd);

      // Handle different modes
      if (highlightMode === 'source') {
        handleSourceEvidence(selectedText, context);
      } else {
        setCurrentSelectedText(selectedText);
        setCurrentContext(context);
        setShowQuestionModal(true);
      }
    }

    // Clear browser selection
    selection.removeAllRanges();
  }, [addHighlight, highlightMode]);

  const handleSourceEvidence = useCallback((selectedText: string, context: string) => {
    if (onSourceEvidence) {
      onSourceEvidence(selectedText, context);
    }
  }, [onSourceEvidence]);

  const handleQuestionSubmit = useCallback((question: string) => {
    if (onHighlight) {
      onHighlight(currentSelectedText, currentContext, question);
    }
    handleModalClose();
  }, [onHighlight, currentSelectedText, currentContext]);

  const handleModalClose = useCallback(() => {
    setShowQuestionModal(false);
    setCurrentSelectedText('');
    setCurrentContext('');
  }, []);

  // Renders a text segment with proper formatting.
  const renderTextSegment = useCallback((segment: TextSegment, index: number) => {
    if (segment.isHighlighted) {
      return (
        <span
          key={`highlight-${segment.highlightId}-${index}`}
          className={`highlighted-text transition-all duration-300 ${
            segment.isNew ? 'animate-pulse' : ''
          }`}
          style={{
            backgroundColor: '#e0f2ff',
            borderRadius: '4px',
            padding: '1px 2px',
            boxShadow: segment.isNew ? '0 0 8px rgba(96, 165, 250, 0.5)' : 'none'
          }}
          title="Highlighted text"
        >
          {segment.text}
        </span>
      );
    }
    return segment.text;
  }, []);

  // Formats text including markdown and highlights.
  const formatText = useCallback((text: string) => {
    if (!text) return '';

    const lines = text.split('\n').filter(line => line.trim() !== '');
    
    return lines.map((line, lineIndex) => {
      const trimmedLine = line.trim();
      
      // Handle headers (### ## #)
      if (trimmedLine.match(/^#{1,3}\s/)) {
        const level = trimmedLine.match(/^(#{1,3})/)?.[1].length || 3;
        const content = trimmedLine.replace(/^#{1,3}\s/, '').trim();
        
        const headerContent = renderTextWithSegments(content);
        
        if (level === 1) {
          return <h3 key={`header-${lineIndex}`} className="text-lg font-bold text-blue-800 mb-3 mt-4">{headerContent}</h3>;
        } else if (level === 2) {
          return <h4 key={`header-${lineIndex}`} className="text-base font-semibold text-blue-700 mb-2 mt-3">{headerContent}</h4>;
        } else {
          return <h5 key={`header-${lineIndex}`} className="text-sm font-medium text-blue-600 mb-2 mt-3">{headerContent}</h5>;
        }
      }

      // Handle bullet points - or * starting
      if (trimmedLine.match(/^\s*[\-\*]\s/)) {
        const content = trimmedLine.replace(/^\s*[\-\*]\s/, '').trim();
        return (
          <div key={`bullet-${lineIndex}`} className="flex items-start mb-2 ml-4">
            <span className="text-blue-600 mr-3 mt-1 font-bold">•</span>
            <span className="flex-1">{renderTextWithSegments(formatInlineText(content))}</span>
          </div>
        );
      }

      // Handle numbered lists 
      if (trimmedLine.match(/^\s*\d+\.\s/)) {
        const number = trimmedLine.match(/^\s*(\d+)\./)?.[1] || '1';
        const content = trimmedLine.replace(/^\s*\d+\.\s/, '').trim();
        return (
          <div key={`numbered-${lineIndex}`} className="flex items-start mb-2 ml-4">
            <span className="text-blue-600 mr-3 mt-1 font-semibold min-w-[1.5rem]">{number}.</span>
            <span className="flex-1">{renderTextWithSegments(formatInlineText(content))}</span>
          </div>
        );
      }

      // Regular paragraph
      return (
        <p key={`para-${lineIndex}`} className="mb-3 leading-relaxed">
          {renderTextWithSegments(formatInlineText(trimmedLine))}
        </p>
      );
    });
  }, [textSegments]);

  // Renders text with applied highlights.
  const renderTextWithSegments = useCallback((content: React.ReactNode): React.ReactNode => {
    if (typeof content !== 'string') {
      return content;
    }

    // Find which segments apply to this content
    const relevantSegments = textSegments.filter(segment => 
      content.includes(segment.text) && segment.text.trim().length > 0
    );

    if (relevantSegments.length === 0) {
      return content;
    }

    let result: React.ReactNode = content;
    
    // Apply highlights to segments
    relevantSegments.forEach((segment, index) => {
      if (typeof result === 'string' && result.includes(segment.text)) {
        const parts = result.split(segment.text);
        const renderedParts: React.ReactNode[] = [];
        
        parts.forEach((part, partIndex) => {
          renderedParts.push(part);
          if (partIndex < parts.length - 1) {
            renderedParts.push(renderTextSegment(segment, index));
          }
        });
        
        result = renderedParts;
      }
    });
    
    return result;
  }, [textSegments, renderTextSegment]);

  // Formats inline text elements like bold, italic
  const formatInlineText = useCallback((text: string): string => {
  
    // Handles bold text **text** or __text__
    const boldRegex = /(\*\*|__)(.*?)\1/g;
    let result = text;
    let match;
    
    while ((match = boldRegex.exec(text)) !== null) {
      result = result.replace(match[0], match[2]);
    }
    
    return result;
  }, []);

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
          {highlightCount > 0 && (
            <button
              onClick={clearHighlights}
              className="text-xs text-red-600 hover:text-red-800 px-2 py-1 rounded border border-red-300 hover:bg-red-50 transition-colors"
              title="Clear all highlights"
            >
              Clear Highlights ({highlightCount})
            </button>
          )}
          <span className="text-sm text-gray-500">
            {highlightMode === 'source' 
              ? 'Highlight text to see source evidence' 
              : 'Highlight text to ask questions'
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

      {/* Question modal */}
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