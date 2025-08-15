import React, { useState, useCallback, useEffect } from "react";
import PDFViewer from './components/PDFViewer';
import AudienceSelector from './components/AudienceSelector';
import SimilarPapersBox from './components/SimilarPapersBox';
import type { AudienceType } from './components/AudienceSelector';
import HighlightableText from './components/HighlightableText';

// Theme context
const ThemeContext = React.createContext<{
  isDark: boolean;
  toggleTheme: () => void;
}>({
  isDark: false,
  toggleTheme: () => {},
});

const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Check system preference initially
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const [isDark, setIsDark] = useState(prefersDark);

  useEffect(() => {
    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setIsDark(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const toggleTheme = () => setIsDark(!isDark);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
  }, [isDark]);

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Theme toggle button.
const ThemeToggle: React.FC = () => {
  const { isDark, toggleTheme } = React.useContext(ThemeContext);
  return (
    <button
      onClick={toggleTheme}
      className="fixed top-4 right-4 p-2 rounded-full transition-all duration-200 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 shadow-sm border border-gray-300 dark:border-gray-600"
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <svg 
        className="w-5 h-5 text-gray-600 dark:text-gray-300" 
        fill="none" 
        viewBox="0 0 24 24" 
        stroke="currentColor"
      >
        {isDark ? (
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" 
          />
        ) : (
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" 
          />
        )}
      </svg>
    </button>
  );
};



// Sidebar for document management.
const Sidebar: React.FC<{ files: string[], onSummarize: (filename: string) => void, summaries: Record<string, string>, isCollapsed: boolean, onFileSelect: (filename: string) => void, selectedFiles: string[], onRemoveFile: (filename: string) => void, onToggleCollapse: () => void } > = ({ files, onSummarize, summaries, isCollapsed, onFileSelect, selectedFiles, onRemoveFile, onToggleCollapse }) => (
  <aside className={`flex-shrink-0 ${isCollapsed ? 'w-16' : 'w-80'} bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 h-full transition-width duration-300 overflow-y-auto flex flex-col border-r border-zinc-200 dark:border-zinc-700`}>
    <div className="p-4 border-b border-zinc-200 dark:border-zinc-700 flex items-center justify-between">
      {!isCollapsed && (
        <div className="flex items-center gap-2">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">Documents</h2>
        </div>
      )}
      {/* Toggle button */}
       <button 
          onClick={onToggleCollapse} 
          className="p-1 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100"
          title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
       >
           {isCollapsed ? '>' : '<'}
       </button>
    </div>

    <ul className="flex-1 overflow-y-auto space-y-3 p-4">
      {files.length === 0 ? (
        <li className={`text-zinc-500 dark:text-zinc-400 italic ${isCollapsed ? 'hidden' : ''}`}>No documents yet.</li>
      ) : (
        files.map((file, idx) => {
          const isSelected = selectedFiles.includes(file);
          const summaryStatus = summaries[file] 
              ? (summaries[file] === "Summarizing..." ? "Summarizing..." : "Summarized") 
              : "Ready";

          const originalFilename = file.replace(/^\d{8}_\d{6}_/, '');
          const filenameParts = file.split('_');
          let uploadDate = 'Unknown Date';
          if (filenameParts.length >= 2) {
              const datePart = filenameParts[0];
              const year = datePart.substring(0, 4);
              const month = datePart.substring(4, 6);
              const day = datePart.substring(6, 8);
              const dateStringUTC = `${year}-${month}-${day}T00:00:00Z`;
              const dateObj = new Date(dateStringUTC);
              
              if (!isNaN(dateObj.getTime())) {
                 uploadDate = dateObj.toLocaleDateString('en-US', { timeZone: 'America/New_York' });
              }
          }

          return (
            <li key={idx} 
                className={`cursor-pointer p-3 rounded-lg shadow-sm transition-colors duration-200 
                  ${isSelected 
                    ? 'bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800' 
                    : 'bg-white dark:bg-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-700 border border-transparent'}`}
                onClick={() => onFileSelect(file)}
            >
              <div className="flex items-start justify-between mb-2 min-h-0">
                <div className="flex-1 min-w-0 mr-2">
                  <span className={`block font-medium text-sm break-words 
                    ${isSelected 
                      ? 'text-blue-700 dark:text-blue-300' 
                      : 'text-zinc-900 dark:text-zinc-100'} 
                    ${isCollapsed ? 'hidden' : ''}`}>
                    {originalFilename}
                  </span>
                </div>
                {!isCollapsed && (
                   <span className={`flex-shrink-0 text-xs font-semibold px-2 py-1 rounded-full whitespace-nowrap
                     ${summaryStatus === "Summarized" 
                       ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                       : summaryStatus === "Summarizing..." 
                       ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'
                       : 'bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300'}`}>
                     {summaryStatus}
                   </span>
                )}
              </div>
               {!isCollapsed && (
                  <div className={`flex justify-between items-center text-xs 
                    ${isSelected 
                      ? 'text-blue-600 dark:text-blue-300' 
                      : 'text-zinc-500 dark:text-zinc-400'}`}>
                     <span>Uploaded: {uploadDate}</span>
                     {summaryStatus === "Ready" && (
                         <button 
                           onClick={(e) => { e.stopPropagation(); onSummarize(file); }} 
                           className="ml-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200"
                         >
                           Summarize
                         </button>
                     )}
                     <button 
                       onClick={(e) => { e.stopPropagation(); onRemoveFile(file); }} 
                       className="ml-2 text-red-500 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 font-medium"
                     >
                       Remove
                     </button>
                  </div>
               )}
            </li>
          );
        })
      )}
    </ul>
  </aside>
);

// PDF upload area.
const PDFUpload: React.FC<{ onUpload: (file: File) => void, uploading: boolean }> = ({ onUpload, uploading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFileName(file.name);
      onUpload(file);
      // Clear the input after selection
      e.target.value = ''; 
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        if (file.type === 'application/pdf') {
            setSelectedFileName(file.name);
            onUpload(file);
        } else {
            alert("Only PDF files are supported.");
        }
    }
  }, [onUpload]);

  return (
    <div className="bg-white dark:bg-zinc-800 rounded-lg shadow-lg p-6 mb-8 border border-zinc-200 dark:border-zinc-700">
      <h3 className="text-xl font-semibold mb-4 text-zinc-900 dark:text-zinc-100">Upload Document</h3>
      
      <div 
        className={`border-2 rounded-lg p-8 text-center transition-all duration-200 
          ${isDragOver 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 border-solid' 
            : 'border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800/50 border-dashed hover:border-blue-400 dark:hover:border-blue-500 hover:bg-zinc-100 dark:hover:bg-zinc-800'}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <label className="block text-sm text-zinc-600 dark:text-zinc-400 cursor-pointer">
          {selectedFileName ? (
            <div className="flex items-center justify-center">
               {uploading ? (
                 <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-600 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                   <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                   <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l2-2.647z"></path>
                 </svg>
               ) : (
                 <svg className="w-6 h-6 text-emerald-500 dark:text-emerald-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
               )}
              <span className="text-zinc-900 dark:text-zinc-100">{selectedFileName}</span>
            </div>
          ) : (
            <>Drag and drop your PDF here, or <span className="text-blue-600 dark:text-blue-400 underline hover:text-blue-800 dark:hover:text-blue-300">browse files</span></>
          )}
          <input type="file" accept="application/pdf" onChange={handleFileChange} disabled={uploading} className="hidden" />
        </label>
      </div>
    </div>
  );
};

// AI Summary display component.
const SummaryDisplay: React.FC<{ summary: string | null, filename: string }> = ({ summary, filename }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [selectedSentence, setSelectedSentence] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [sourceEvidence, setSourceEvidence] = useState<any>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const [loadingSourceEvidence, setLoadingSourceEvidence] = useState(false);
  const [showExplanationResult, setShowExplanationResult] = useState(false);
  const [showSourceEvidenceResult, setShowSourceEvidenceResult] = useState(false);

  // Handles text highlighting and fetching explanations.
  const handleTextHighlight = async (selectedText: string, context: string, question: string) => {
    setSelectedSentence(selectedText);
    setLoadingExplanation(true);
    setShowExplanationResult(true);
    setShowSourceEvidenceResult(false); // Hide source evidence when showing explanation
    
    try {
      const response = await fetch(`${BACKEND_URL}/explain-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          filename, 
          selected_text: selectedText,
          context,
          question,
          //Default = patient
          audience_type: 'patient' 
        })
      });
      
      if (!response.ok) throw new Error('Failed to fetch explanation');
      const data = await response.json();
      setExplanation({ ...data, userQuestion: question }); 
    } catch (error) {
      console.error('Error fetching explanation:', error);
      setExplanation({ 
        explanation: 'Failed to get explanation. Please try again.',
        userQuestion: question 
      });
    } finally {
      setLoadingExplanation(false);
    }
  };

  // Handles fetching source evidence for highlighted text.
  const handleSourceEvidence = async (selectedText: string) => {
    setSelectedSentence(selectedText);
    setLoadingSourceEvidence(true);
    setShowSourceEvidenceResult(true);
    
    setShowExplanationResult(false); 
    
    try {
      const response = await fetch(`${BACKEND_URL}/explanation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          filename, 
          sentence: selectedText
        })
      });
      
      if (!response.ok) throw new Error('Failed to fetch source evidence');
      const data = await response.json();
      setSourceEvidence(data);
    } catch (error) {
      console.error('Error fetching source evidence:', error);
      setSourceEvidence({ 
        source_chunks: [], 
        confidence: 0,
        error: 'Failed to get source evidence. Please try again.' 
      });
    } finally {
      setLoadingSourceEvidence(false);
    }
  };

  // Show a loading indicator if summary is in progress
  if (summary === "Summarizing...") {
      return (
          <div className="bg-white dark:bg-zinc-800 rounded-lg shadow-lg p-6 mb-8 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center">
               <svg className="animate-spin h-8 w-8 text-blue-600 dark:text-blue-400 mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l2-2.647z"></path>
               </svg>
               <span className="text-lg text-zinc-900 dark:text-zinc-100">Generating Summary...</span>
          </div>
      );
  }

  if (!summary) {
    return null;
  }
  
  const handleCopyToClipboard = () => {
      navigator.clipboard.writeText(summary).then(() => {
          alert('Summary copied to clipboard!'); 
      }).catch(err => {
          console.error('Failed to copy summary: ', err);
      });
  };

  return (
     <div className="bg-white dark:bg-zinc-800 rounded-lg shadow-lg p-6 mb-8 border border-zinc-200 dark:border-zinc-700">
       <div className="flex justify-between items-center mb-4 cursor-pointer" onClick={() => setIsCollapsed(!isCollapsed)}>
         <h3 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">AI Summary</h3>
         <button className="text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200">
             {isCollapsed ? '+' : '-'}
         </button>
       </div>
      
       {!isCollapsed && (
           <div className="text-zinc-700 dark:text-zinc-300">
             <HighlightableText 
               text={summary} 
               onHighlight={handleTextHighlight}
               onSourceEvidence={handleSourceEvidence}
               className="leading-relaxed"
             />
           </div>
       )}

      {/* Source Evidence Result */}
      {showSourceEvidenceResult && (
        <div className="mt-4 p-4 bg-emerald-50 dark:bg-emerald-900/10 rounded-lg border border-emerald-200 dark:border-emerald-800">
          <div className="flex justify-between items-start mb-3">
            <h4 className="text-sm font-semibold text-emerald-800 dark:text-emerald-300">Source Evidence</h4>
            <button
              onClick={() => {
                setShowSourceEvidenceResult(false);
                setSourceEvidence(null);
              }}
              className="text-emerald-600 dark:text-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 text-sm"
            >
              Close
            </button>
          </div>
          {loadingSourceEvidence ? (
            <div className="text-sm text-emerald-700 dark:text-emerald-300">Finding source evidence...</div>
          ) : sourceEvidence?.error ? (
            <div className="text-sm text-red-600 dark:text-red-400">{sourceEvidence.error}</div>
          ) : (
            <>
              <div className="mb-3 p-2 bg-white dark:bg-zinc-800 rounded border border-emerald-200 dark:border-emerald-800">
                <div className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-1">Highlighted Text:</div>
                <div className="text-sm text-zinc-700 dark:text-zinc-300 italic">
                  "{selectedSentence && selectedSentence.length > 200 
                    ? `${selectedSentence.substring(0, 200)}...` 
                    : selectedSentence}"
                </div>
              </div>
              
              <div className="mb-3 flex items-center gap-2">
                <span className="text-sm font-medium text-emerald-800 dark:text-emerald-300">
                  Overall Confidence: {((sourceEvidence?.confidence || 0) * 100).toFixed(2)}%
                </span>
              </div>
              
              <div className="space-y-3">
                {sourceEvidence?.source_chunks?.map((chunk: any, idx: number) => (
                  <div key={idx} className="p-3 bg-white dark:bg-zinc-800 rounded border border-emerald-200 dark:border-emerald-800">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
                          Source {idx + 1} (Similarity: {(chunk.similarity * 100).toFixed(2)}%)
                        </span>
                      </div>
                      {chunk.metadata?.page && (
                        <span className="text-xs text-zinc-500 dark:text-zinc-400">
                          Page {chunk.metadata.page}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-zinc-800 dark:text-zinc-200 leading-relaxed">
                      {chunk.content.length > 350 
                        ? `${chunk.content.substring(0, 350)}...` 
                        : chunk.content
                      }
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Explanation Result */}
      {showExplanationResult && explanation && (
        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/10 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="flex justify-between items-start mb-3">
            <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-300">AI Explanation</h4>
            <button
              onClick={() => {
                setShowExplanationResult(false);
                setExplanation(null);
              }}
              className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200 text-sm"
            >
              Close
            </button>
          </div>
          {loadingExplanation ? (
            <div className="text-sm text-blue-700 dark:text-blue-300">Loading explanation...</div>
          ) : (
            <>
              {explanation.userQuestion && (
                <div className="mb-3 p-2 bg-white dark:bg-zinc-800 rounded border border-blue-200 dark:border-blue-800">
                  <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1">Your Question:</div>
                  <div className="text-sm text-zinc-800 dark:text-zinc-200 italic">"{explanation.userQuestion}"</div>
                </div>
              )}
              
              <div className="mb-3 p-2 bg-white dark:bg-zinc-800 rounded border border-blue-200 dark:border-blue-800">
                <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1">Selected Text:</div>
                <div className="text-sm text-zinc-700 dark:text-zinc-300 italic">
                  "{selectedSentence && selectedSentence.length > 200 
                    ? `${selectedSentence.substring(0, 200)}...` 
                    : selectedSentence}"
                </div>
              </div>
              
              <div className="p-3 bg-white dark:bg-zinc-800 rounded border border-blue-200 dark:border-blue-800">
                <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-2">AI Response:</div>
                <div className="text-sm text-zinc-800 dark:text-zinc-200 leading-relaxed">
                  {explanation.explanation}
                </div>
              </div>
            </>
          )}
        </div>
      )}
      
       {/* Summary action buttons */}
       <div className="mt-4 flex justify-end">
           <button 
             onClick={handleCopyToClipboard} 
             className="px-4 py-2 bg-blue-500 dark:bg-blue-600 text-white rounded hover:bg-blue-600 dark:hover:bg-blue-700 transition-colors duration-200 text-sm"
           >
             Copy Summary
           </button>
       </div>
    </div>
  );
};

// Chat interface.
const Chat: React.FC<{ files: string[] }> = ({ files }) => {
  const [messages, setMessages] = useState<{ sender: "user" | "bot", text: string, sources?: any[] }[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Clear chat history
  const handleClearChat = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/chat/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      
      if (res.ok) {
        setMessages([]);
        console.log("Chat history cleared successfully");
      } else {
        console.warn("Failed to clear chat history on server, clearing locally only");
        setMessages([]);
      }
    } catch (err) {
      console.error("Error clearing chat:", err);
      // Clear locally even if server request fails
      setMessages([]);
    }
  };

  // Format chat messages with proper text handling
  const formatChatMessage = (text: string): React.ReactNode => {
    if (!text) return '';
    
    // Split by paragraphs and format each
    const paragraphs = text.split('\n').filter(p => p.trim() !== '');
    
    return paragraphs.map((paragraph, idx) => {
      // Handle bold text
      const formattedText = paragraph.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      
      return (
        <p key={idx} className={`${idx > 0 ? 'mt-2' : ''} leading-relaxed`}>
          <span dangerouslySetInnerHTML={{ __html: formattedText }} />
        </p>
      );
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || files.length === 0) return;

    const userMessage = { sender: "user" as const, text: question };
    setMessages(prev => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      // Use the chat endpoint for conversational RAG
      const chatHistory = messages.map(msg => ({
        [msg.sender === "user" ? "human" : "ai"]: msg.text
      }));

      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          filenames: files,
          chat_history: chatHistory.length > 0 ? chatHistory : null
        }),
      });

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      
      const botMessage = { 
        sender: "bot" as const, 
        text: data.answer, 
        sources: data.sources 
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      console.error("Chat error:", err);
      const errorMessage = { sender: "bot" as const, text: `Error: ${err}. Please try again.` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any); 
    }
  };

  const suggestedQuestions = [
      "What are the key objectives of this paper?",
      "Summarize the methodology used.",
      "What were the major findings?",
      "Discuss the limitations mentioned.",
      "What is the field of study?",
      "Can you suggest related papers?"
  ];
  const placeholderText = `Ask: ${suggestedQuestions[Math.floor(Math.random() * suggestedQuestions.length)]}`;

  return (
    <div className="bg-white dark:bg-zinc-800 rounded-lg shadow-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 flex flex-col h-full">
       <div className="flex justify-between items-center p-6 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-center gap-4">
            <h3 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">Chat Assistant</h3>
            {messages.length > 0 && !isCollapsed && (
              <button
                onClick={handleClearChat}
                className="px-3 py-1 text-xs text-zinc-600 dark:text-zinc-400 hover:text-red-600 dark:hover:text-red-400 border border-zinc-300 dark:border-zinc-600 hover:border-red-300 dark:hover:border-red-500 rounded-md transition-colors duration-200"
                title="Clear conversation history"
              >
                Clear Chat
              </button>
            )}
          </div>
          <button 
            className="text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? '+' : '-'}
          </button>
       </div>
      
       {!isCollapsed && (
         <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-zinc-50 dark:bg-zinc-900">
           {messages.length === 0 ? (
             <div className="text-zinc-600 dark:text-zinc-400 text-center italic mt-10">Ask a research question about your uploaded papers.</div>
           ) : (
             messages.map((msg, idx) => (
               <div key={idx} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
                 <div className={`inline-block p-4 rounded-xl max-w-sm break-words 
                   ${msg.sender === "user" 
                     ? "bg-blue-600 dark:bg-blue-700 text-white rounded-br-none" 
                     : "bg-zinc-200 dark:bg-zinc-700 text-zinc-800 dark:text-zinc-100 rounded-bl-none"}`}>
                   <div className="text-sm leading-relaxed">
                     {formatChatMessage(msg.text)}
                   </div>
                   {msg.sender === "bot" && msg.sources && msg.sources.length > 0 && (
                     <div className="mt-3 pt-3 border-t border-zinc-300 dark:border-zinc-600">
                       <div className="text-xs font-medium mb-2 text-zinc-100 dark:text-zinc-300">Sources:</div>
                       {msg.sources.map((source: any, idx: number) => (
                         <div key={idx} className="text-xs bg-white/10 dark:bg-black/10 rounded p-2 mb-1">
                           <span className="font-medium">[{source.index || idx + 1}]</span>
                           {source.chunk ? (
                             <>
                               <span className="text-zinc-200 dark:text-zinc-300"> Section: {source.chunk.metadata?.section || 'Unknown'}</span>
                               <div className="mt-1 text-zinc-300 dark:text-zinc-400">{source.chunk.content.substring(0, 100)}...</div>
                             </>
                           ) : (
                             <span className="text-zinc-300 dark:text-zinc-400"> {source.metadata?.filename || source.content?.substring(0, 100) || 'Unknown source'}</span>
                           )}
                         </div>
                       ))}
                     </div>
                   )}
                 </div>
               </div>
             ))
           )}
           {loading && (
              <div className="flex justify-start">
                  <div className="inline-block p-4 rounded-xl bg-zinc-200 dark:bg-zinc-700 text-zinc-800 dark:text-zinc-100 rounded-bl-none">
                      Loading...
                  </div>
              </div>
           )}
         </div>
       )}

       {!isCollapsed && (
         <form onSubmit={handleSubmit} className="flex gap-4 p-4 border-t border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800">
           <textarea
             value={question}
             onChange={(e) => setQuestion(e.target.value)}
             onKeyDown={handleKeyDown}
             placeholder={placeholderText}
             className="flex-1 border border-zinc-300 dark:border-zinc-600 rounded-lg px-4 py-3 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow duration-200 resize-none placeholder-zinc-500 dark:placeholder-zinc-400 bg-white dark:bg-zinc-800"
             disabled={loading || files.length === 0}
             rows={1}
           />
           <button type="submit" className="flex-shrink-0 bg-blue-700 dark:bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-800 dark:hover:bg-blue-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading || files.length === 0}>
             Send
           </button>
         </form>
       )}
    </div>
  );
};

const BACKEND_URL = import.meta.env.PROD 
  ? "https://medscope.onrender.com"  // Production backend URL
  : "http://localhost:8000";         // Development backend URL

const App: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [summaries, setSummaries] = useState<Record<string, string>>({});
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  // State for sidebar collapse
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false); 
  const [selectedAudience, setSelectedAudience] = useState<AudienceType>('clinician');
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [relatedDocuments, setRelatedDocuments] = useState<any[]>([]);


  const handleUpload = async (file: File) => {
    console.log(`Starting upload for file: ${file.name} to ${BACKEND_URL}/upload`);
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      
      console.log(`Upload response status: ${res.status}`);
      
      if (!res.ok) {
        const errorText = await res.text();
        console.error('Upload failed:', errorText);
        throw new Error(`HTTP error! status: ${res.status} - ${errorText}`);
      }
      
      const data = await res.json();
      console.log('Upload successful, response data:', data);
      
      setUploadedFiles((prev) => [...prev, data.filename]);

      // Clear summary if file is re-uploaded or new file is uploaded
      setSummaries(prev => { delete prev[data.filename]; return { ...prev }; });
      // Select the newly uploaded file
      setSelectedFiles([data.filename]); 
      
      // Fetch related documents
      fetchRelatedDocuments(data.filename);
    } catch (err) {
      console.error("Upload error:", err);
      alert(`Failed to upload PDF: ${err}`); 
    } finally {
      setUploading(false);
    }
  };

  const fetchRelatedDocuments = async (filename: string) => {
    try {
      const response = await fetch(`${BACKEND_URL}/related-documents/${filename}`);
      if (response.ok) {
        const data = await response.json();
        setRelatedDocuments(data.related || []);
      }
    } catch (error) {
      console.error('Error fetching related documents:', error);
    }
  };

  const handleSummarize = async (filename: string) => {
     // Prevent summarizing if already summarizing or summary exists
     if (summaries[filename] && summaries[filename] !== "Summarizing...") return;

     setSummaries(prev => ({ ...prev, [filename]: "Summarizing..." }));

    try {
      const res = await fetch(`${BACKEND_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          filename,
          audience_type: selectedAudience 
        }),
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setSummaries((prev) => ({ ...prev, [filename]: data.message }));
      

    } catch (err) {
      console.error("Summarization error:", err);
      setSummaries(prev => ({ ...prev, [filename]: `Summarization failed: ${err}` })); 
    }
  };

   // file removal
  // @ts-ignore
  const handleRemoveFile = async (filename: string) => {
     try {
         const res = await fetch(`${BACKEND_URL}/delete_file`, {
             method: "POST",
             headers: { "Content-Type": "application/json" },
             body: JSON.stringify({ filename }),
         });
         if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

         // Remove from frontend state after successful backend deletion.
         setUploadedFiles(prev => prev.filter(file => file !== filename));
         setSummaries(prev => { delete prev[filename]; return { ...prev }; });
         if (selectedFiles.includes(filename)) {
             setSelectedFiles(prev => prev.filter(file => file !== filename));
         }

         alert(`File deleted successfully: ${filename}`); 
     } catch (err) {
         console.error("File deletion error:", err);
         alert(`Failed to delete file: ${err}`);
     }
   };

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  const handleFileSelect = (filename: string) => {
    // Toggle selection
    setSelectedFiles(prev => 
      prev.includes(filename) 
        ? prev.filter(f => f !== filename)
        : [...prev, filename]
    );
  };

  return (
    <ThemeProvider>
      <div className="flex h-screen bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100">
        {/* Sidebar */}
        <Sidebar 
          files={uploadedFiles} 
          onSummarize={handleSummarize} 
          summaries={summaries}
          isCollapsed={isSidebarCollapsed}
          onFileSelect={handleFileSelect}
          selectedFiles={selectedFiles}
          onRemoveFile={handleRemoveFile}
          onToggleCollapse={toggleSidebar}
        />

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col p-8 overflow-hidden bg-white dark:bg-zinc-900">
          {/* Header with Title and Tagline */}
          <header className="mb-8">
            <h1 className="text-5xl font-extrabold text-blue-900 dark:text-blue-400 mb-2">MedPub</h1>
            <p className="text-xl text-zinc-600 dark:text-zinc-400">Your AI and ML-powered assistant for exploring medical research with precision and depth</p>
          </header>

          {/* Content area: Upload, Summary, and Chat in a single column */}
          <div className="flex-1 flex flex-col gap-8 overflow-y-auto">
             {/* Upload component always visible */}
             <PDFUpload onUpload={handleUpload} uploading={uploading} />
                
             {/* Audience Selector */}
             {selectedFiles.length > 0 && (
               <AudienceSelector
                 selectedAudience={selectedAudience}
                 onAudienceChange={setSelectedAudience}
                 disabled={Object.values(summaries).some(s => s === "Summarizing...")}
               />
             )}
             
             {/* Container for Summary and Chat*/}
             {selectedFiles.length > 0 ? (
               <div className="flex-1 flex flex-col gap-8">
                  {false && (
                    <div className="flex justify-between items-center">
                      <button
                        onClick={() => setShowPdfViewer(!showPdfViewer)}
                        className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md transition-colors"
                      >
                        {showPdfViewer ? 'Hide PDF Viewer' : 'Show PDF Viewer'}
                      </button>
                      {/* Related Documents */}
                      {relatedDocuments.length > 0 && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-sm font-semibold text-gray-700 mb-3">Related Documents</h4>
                          <div className="space-y-2">
                            {relatedDocuments.slice(0, 3).map((doc, idx) => (
                              <div key={idx} className="flex items-start gap-3 p-2 bg-white rounded border border-gray-200">
                                <div className="flex-1">
                                  <div className="text-sm font-medium text-blue-600 truncate">
                                    {doc.title || doc.filename.replace(/^[\d]{8}_[\d]{6}_/, '')}
                                  </div>
                                  <div className="text-xs text-gray-500 mt-1">
                                    Similarity: {(doc.similarity_score * 100).toFixed(1)}% • 
                                    Common topics: {doc.common_topics?.join(', ') || 'None'}
                                  </div>
                                </div>
                                <div className="text-xs text-gray-400">
                                  #{idx + 1}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {false && showPdfViewer && selectedFiles.length === 1 && (
                    <div className="h-96 border border-gray-300 rounded-lg overflow-hidden">
                      <PDFViewer
                        filename={selectedFiles[0]}
                        backendUrl={BACKEND_URL}
                      />
                    </div>
                  )}

                  {/* Similar Research Papers Box */}
                  {selectedFiles.length === 1 && (
                    <SimilarPapersBox 
                      filename={selectedFiles[0]}
                      backendUrl={BACKEND_URL}
                      onError={(error) => {
                        console.error('Similar papers error:', error);
                      }}
                    />
                  )}

                  {/* Displays summary if one file is selected. */}
                  {selectedFiles.length === 1 && summaries[selectedFiles[0]] && summaries[selectedFiles[0]] !== "Summarizing..." && (
                     <SummaryDisplay 
                       summary={summaries[selectedFiles[0]]} 
                       filename={selectedFiles[0]}
                     />
                  )}
                  <div className="flex-1 h-full min-h-[400px]">
                     <Chat files={selectedFiles} />{} 
                  </div>
               </div>
               ) : (
                  // Welcome/Instructional Message when no file is selected
                  <div className="flex-1 flex flex-col items-center justify-center text-center text-zinc-600 dark:text-zinc-400 italic p-8 border-2 border-dashed border-zinc-300 dark:border-zinc-600 rounded-lg bg-white dark:bg-zinc-800">
                      <p className="text-2xl font-semibold mb-2">Welcome to MedPub!</p>
                      <p>Upload a medical paper to get started. You can drag and drop a PDF above or click to browse your files.</p>
                      <p className="mt-4 text-sm">Once uploaded, MedPub applies natural language understanding to deliver summaries, retrieve similar research, map AI outputs to source text for interpretability, and support real-time contextual Q&A.</p>
                  </div>
              )}
          </div>
        </main>
        <ThemeToggle />
      </div>
    </ThemeProvider>
  );
};

export default App;
