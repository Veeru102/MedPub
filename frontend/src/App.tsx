import React, { useState, useCallback, useMemo } from "react";

// Placeholder component for the PDF viewer area - Keep for future use
// @ts-ignore
const PdfViewerPlaceholder: React.FC = () => (
    <div className="w-full h-96 bg-gray-100 flex items-center justify-center text-gray-500 italic rounded-md border border-dashed border-gray-300">
        PDF Preview Area (Implementation Needed)
        {/* TODO: Integrate a PDF rendering library like PDF.js here. */}
        {/* You would fetch the PDF content (or a URL) and render it in this div. */}
    </div>
);

// Helper component for the collapsible sidebar - Now displays interactive document cards
const Sidebar: React.FC<{ files: string[], onSummarize: (filename: string) => void, summaries: Record<string, string>, isCollapsed: boolean, onFileSelect: (filename: string) => void, selectedFiles: string[], onRemoveFile: (filename: string) => void, onToggleCollapse: () => void } > = ({ files, onSummarize, summaries, isCollapsed, onFileSelect, selectedFiles, onRemoveFile, onToggleCollapse }) => (
  <aside className={`flex-shrink-0 ${isCollapsed ? 'w-16' : 'w-80'} bg-gray-800 text-gray-200 h-full transition-width duration-300 overflow-y-auto flex flex-col border-r border-blue-900`}>
    <div className="p-4 border-b border-blue-900 flex items-center justify-between">
      {!isCollapsed && <h2 className="text-xl font-semibold text-blue-300">Documents</h2>}
      {/* Toggle Button */}
       <button 
          onClick={onToggleCollapse} 
          className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white"
          title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
       >
           {/* Simple text toggle, replace with icon later */}
           {isCollapsed ? '>' : '<'}
       </button>
    </div>
    {/* Search and Filter Placeholder */}
    {/* {!isCollapsed && <input type="text" placeholder="Search..." className="m-4 p-2 rounded bg-gray-700 text-gray-200 border border-gray-600 focus:outline-none focus:border-blue-500" />} */}

    <ul className="flex-1 overflow-y-auto space-y-3 p-4">
      {files.length === 0 ? (
        <li className={`text-gray-400 italic ${isCollapsed ? 'hidden' : ''}`}>No documents yet.</li>
      ) : (
        files.map((file, idx) => {
          const isSelected = selectedFiles.includes(file);
          // Determine summary status: Summarizing, Summarized, or Ready
          const summaryStatus = summaries[file] 
              ? (summaries[file] === "Summarizing..." ? "Summarizing..." : "Summarized") 
              : "Ready";

          // Extract original filename by removing timestamp prefix (YYYYMMDD_HHMMSS_)
          const originalFilename = file.replace(/^\d{8}_\d{6}_/, '');

          // Parse date from filename format YYYYMMDD_HHMMSS
          const filenameParts = file.split('_');
          let uploadDate = 'Unknown Date';
          if (filenameParts.length >= 2) {
              const datePart = filenameParts[0]; // YYYYMMDD
              const year = datePart.substring(0, 4);
              const month = datePart.substring(4, 6);
              const day = datePart.substring(6, 8);
              // Create a date string like YYYY-MM-DD to parse
              const dateString = `${year}-${month}-${day}`;
              const dateObj = new Date(dateString);
              if (!isNaN(dateObj.getTime())) {
                 uploadDate = dateObj.toLocaleDateString(); // Format as local date string
              }
          }

          return (
            <li key={idx} 
                className={`cursor-pointer p-3 rounded-lg shadow-md transition-colors duration-200 ${isSelected ? 'bg-blue-700 border border-blue-500' : 'bg-gray-700 hover:bg-gray-600 border border-transparent'}`}
                onClick={() => onFileSelect(file)} // Toggle selection for the file
            >
              <div className="flex justify-between items-center mb-2">
                {/* Suggestion: Add a document icon */} 
                <span className={`flex-1 truncate font-medium ${isSelected ? 'text-white' : 'text-blue-400'} ${isCollapsed ? 'hidden' : 'mr-2'}`}>{originalFilename}</span> {/* Use original filename here */}
                {!isCollapsed && (
                   <span className={`flex-shrink-0 text-xs font-semibold px-2 py-1 rounded-full
                     ${summaryStatus === "Summarized" ? 'bg-green-600 text-white'
                       : summaryStatus === "Summarizing..." ? 'bg-yellow-600 text-white'
                       : 'bg-gray-500 text-white'}`}>
                     {summaryStatus}
                   </span>
                )}
              </div>
               {!isCollapsed && (
                  <div className={`flex justify-between items-center text-xs ${isSelected ? 'text-blue-200' : 'text-gray-400'}`}>
                     <span>Uploaded: {uploadDate}</span>
                     {summaryStatus === "Ready" && (
                         <button onClick={(e) => { e.stopPropagation(); onSummarize(file); }} className="ml-2 text-blue-300 hover:text-blue-100">Summarize</button>
                     )}
                     {/* Ensure red text for remove and add hover effect */}
                     <button onClick={(e) => { e.stopPropagation(); onRemoveFile(file); }} className="ml-2 text-red-400 hover:text-red-600 font-medium">Remove</button>
                  </div>
               )}
              {/* Summary preview (optional, could add a small snippet here if not collapsed) */}
              {/* {summaries[file] && !isCollapsed && summaryStatus === "Summarized" && <p className="text-xs text-gray-400 mt-2 line-clamp-2">{summaries[file]}</p>} */}
            </li>
          );
        })
      )}
    </ul>
    {/* + Upload Floating Button (adjust positioning as needed) */}
    {/* This could trigger a modal or scroll to the upload section */}
    {/* {!isCollapsed && (
        <div className="p-4">
            <button className="w-full bg-blue-600 text-white py-2 rounded-full flex items-center justify-center shadow-lg hover:bg-blue-700 transition-colors duration-200">
                 + Upload
            </button>
        </div>
    )} */}
     {/* Sizing Adjust: Adding a draggable sizing handle is a complex feature and is not implemented in this step. */}
  </aside>
);

// Helper component for PDF upload area with drag-and-drop
const PDFUpload: React.FC<{ onUpload: (file: File) => void, uploading: boolean }> = ({ onUpload, uploading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFileName(file.name);
      onUpload(file);
      e.target.value = ''; // Clear the input after selection
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
            alert("Only PDF files are supported."); // Suggestion: Use a better notification
        }
    }
  }, [onUpload]);

  return (
    // Card container
    <div className="bg-white rounded-lg shadow-lg p-6 mb-8 border border-blue-200">
      <h3 className="text-xl font-semibold mb-4 text-blue-800">Upload Document</h3>
      
      <div 
        className={`border-2 rounded-lg p-8 text-center transition-all duration-200 
          ${isDragOver 
            ? 'border-blue-500 bg-blue-50 border-solid' 
            : 'border-gray-300 bg-gray-50 border-solid hover:border-blue-400 hover:bg-gray-100'}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <label className="block text-sm text-gray-600 cursor-pointer">
          {selectedFileName ? (
            <div className="flex items-center justify-center">
               {uploading ? (
                 <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l2-2.647z"></path>
                 </svg>
               ) : (
                 <svg className="w-6 h-6 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
               )}
              <span>{selectedFileName}</span>
            </div>
          ) : (
            <>Drag and drop your PDF here, or <span className="text-blue-600 underline">browse files</span></>
          )}
          <input type="file" accept="application/pdf" onChange={handleFileChange} disabled={uploading} className="hidden" />
        </label>
      </div>
    </div>
  );
};

// Helper component for AI Summary display
const SummaryDisplay: React.FC<{ summary: string | null }> = ({ summary }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Define the specific headings to format
  const headingsToFormat = [
      "Limitations:",
      "Major Findings:",
      "Methodology:",
      "Summary of the Paper:",
      "Key Objectives:",
      "Suggested Related Papers:"
  ];

  // Advanced parsing for sections and stripping markdown
  const parseAndFormatSummary = (text: string) => {
      const sections: Array<{ title: string | null, content: string }> = [];
      const lines = text.split('\n');
      let currentTitle: string | null = null;
      let currentContent: string[] = [];

      const addSection = () => {
          if (currentTitle !== null || currentContent.length > 0) {
              sections.push({
                  title: currentTitle ? currentTitle.replace(/^#+\s*/, '').replace(/\*\*/g, '') : null, // Remove markdown headings and **
                  content: currentContent.join('\n').trim()//.replace(/^\*\*(.*?)\*\*:\s*/, '') // Remove bold markdown and leading colon/space from content start - REMOVED
              });
          }
          currentTitle = null;
          currentContent = [];
      };

      for (const line of lines) {
          const trimmedLine = line.trim();

          // Check for markdown headings (###, ##, #) or bold headings followed by colon
          const boldHeadingMatch = trimmedLine.match(/^\*\*(.*?)\*\*:\s*/);
          const markdownHeadingMatch = trimmedLine.match(/^#+\s(.*)/);

          if (boldHeadingMatch) {
              addSection(); // Add previous section
              currentTitle = boldHeadingMatch[1]; // Capture text inside **
              currentContent = [trimmedLine.substring(boldHeadingMatch[0].length).trim()]; // Start content after the bold heading
          } else if (markdownHeadingMatch) {
               addSection(); // Add previous section
               currentTitle = markdownHeadingMatch[1].trim(); // Capture text after markdown #
          } else if (trimmedLine) { // Non-empty line
              currentContent.push(trimmedLine);
          } else if (currentContent.length > 0) { // Empty line as a potential paragraph break within content
              currentContent.push(''); // Preserve empty lines to some extent for spacing
          }
      }
      addSection(); // Add the last section

       // Fallback if no sections were parsed but there was text
       if (sections.length === 0 && text.trim()) {
           sections.push({ title: null, content: text.trim().replace(/\*\*/g, '') }); // Remove ** from fallback content too
       }

      return sections;
  };

  const summarySections = useMemo(() => (summary ? parseAndFormatSummary(summary) : []), [summary]);

  // Show a loading indicator if summary is in progress
  if (summary === "Summarizing...") {
      return (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8 border border-blue-200 flex items-center justify-center">
               <svg className="animate-spin h-8 w-8 text-blue-600 mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l2-2.647z"></path>
               </svg>
               <span className="text-lg text-blue-800">Generating Summary...</span>
          </div>
      );
  }

  if (!summary) {
    return null; // Don't display card if no summary
  }
  
  const handleCopyToClipboard = () => {
      navigator.clipboard.writeText(summary).then(() => {
          alert('Summary copied to clipboard!'); // Suggestion: Use a better toast/notification
      }).catch(err => {
          console.error('Failed to copy summary: ', err);
      });
  };

  return (
     // Card container
    <div className="bg-white rounded-lg shadow-lg p-6 mb-8 border border-blue-200">
       <div className="flex justify-between items-center mb-4 cursor-pointer" onClick={() => setIsCollapsed(!isCollapsed)}>
         <h3 className="text-xl font-semibold text-blue-800">AI Summary</h3>
         {/* Suggestion: Add collapse/expand icon */}
          <button className="text-gray-500 hover:text-gray-700">
              {isCollapsed ? '+' : '-'}
          </button>
       </div>
      
       {!isCollapsed && (
           <div className="text-gray-700">
             {summarySections.map((section, index) => (
               <div key={index} className="mb-4">
                 {section.title && (
                   <h4 className={`${headingsToFormat.includes(section.title) ? 'text-xl font-bold text-blue-800' : 'text-lg font-semibold'} mb-2`}>{section.title}</h4>
                 )}
                 {section.content.split('\n').map((paragraph, pIdx) => (
                   <p key={pIdx} className="mb-2 last:mb-0">{paragraph}</p>
                 ))}
               </div>
             ))}
           </div>
       )}

        <button 
           onClick={handleCopyToClipboard} 
           className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors duration-200 text-sm"
        >
           Copy Summary
        </button>
    </div>
  );
};

// Helper component for the Chat interface - Now in a card
const Chat: React.FC<{ files: string[] }> = ({ files }) => {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<{ text: string; sender: "user" | "bot"; sources?: Array<{ content: string; metadata: any }> }[]>([]);
  const [loading, setLoading] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || files.length === 0) {
      // alert("Please upload a document first."); // Use a better notification
      return;
    }

    setLoading(true);
    setMessages((prev) => [...prev, { text: question, sender: "user" }]);
    setQuestion("");

    try {
      const res = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question, filenames: files }),
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          text: data.message,
          sender: "bot",
          sources: data.sources // Store the source documents
        }
      ]);
    } catch (err) {
      console.error("Chat query error:", err);
      setMessages((prev) => [...prev, { text: `Failed to get an answer: ${err}`, sender: "bot" }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any); // Cast needed for form submission type
    }
  };

  // More suggested questions for placeholder
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
     // Card container
    <div className="bg-white rounded-lg shadow-lg overflow-hidden border border-blue-200 flex flex-col h-full">
       <div className="flex justify-between items-center p-6 border-b border-gray-200 cursor-pointer" onClick={() => setIsCollapsed(!isCollapsed)}>
          <h3 className="text-xl font-semibold text-blue-800">Chat Assistant</h3>
           {/* Style the collapse (-) button to be visually minimal and aligned top-right */}
          <button className="text-gray-500 hover:text-gray-800 float-right">
              {isCollapsed ? '+' : '-'}
          </button>
       </div>
      
       {!isCollapsed && (
         <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
           {messages.length === 0 ? (
             <div className="text-gray-600 text-center italic mt-10">Ask a research question about your uploaded papers.</div>
           ) : (
             messages.map((msg, idx) => (
               <div key={idx} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
                 {/* Suggestion: Use avatar/icon for sender */}
                 <div className={`inline-block p-4 rounded-xl max-w-sm break-words ${msg.sender === "user" ? "bg-blue-600 text-white rounded-br-none" : "bg-gray-200 text-gray-800 rounded-bl-none"}`}>
                   {/* Suggestion: Add hover effects or tooltips for longer messages */}
                   {msg.text}
                   {msg.sender === "bot" && msg.sources && msg.sources.length > 0 && (
                     <div className="mt-2 text-xs text-gray-600 italic">
                       Sources: {msg.sources.map(source => source.metadata.filename || "Unknown").join(", ")}
                     </div>
                   )}
                 </div>
               </div>
             ))
           )}
           {loading && (
              <div className="flex justify-start">
                  <div className="inline-block p-4 rounded-xl bg-gray-200 text-gray-800 rounded-bl-none">
                      {/* Simple loading indicator */}
                      Loading...
                  </div>
              </div>
           )}
         </div>
       )}

       {!isCollapsed && (
         <form onSubmit={handleSubmit} className="flex gap-4 p-4 border-t border-gray-200 bg-white">
           {/* Use textarea for multiline input */}
           <textarea
             value={question}
             onChange={(e) => setQuestion(e.target.value)}
             onKeyDown={handleKeyDown} // Handle Shift+Enter
             placeholder={placeholderText} // Use dynamic placeholder
             className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow duration-200 resize-none placeholder-gray-500 bg-white" // Improved text/background contrast and explicit background
             disabled={loading || files.length === 0}
             rows={1} // Start with one row, will expand with content
           />
           <button type="submit" className="flex-shrink-0 bg-blue-700 text-white px-8 py-3 rounded-lg hover:bg-blue-800 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading || files.length === 0}>
             Send
           </button>
         </form>
       )}
    </div>
  );
};

const BACKEND_URL = "https://medscope.onrender.com";

const App: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [summaries, setSummaries] = useState<Record<string, string>>({});
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setUploadedFiles((prev) => [...prev, data.filename]);
      // Clear summary if file is re-uploaded or new file is uploaded
      setSummaries(prev => { delete prev[data.filename]; return { ...prev }; });
      setSelectedFiles([data.filename]); // Select the newly uploaded file
    } catch (err) {
      console.error("Upload error:", err);
      alert(`Failed to upload PDF: ${err}`); // Suggestion: Use a better notification system
    } finally {
      setUploading(false);
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
        body: JSON.stringify({ filename }),
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setSummaries((prev) => ({ ...prev, [filename]: data.message }));
    } catch (err) {
      console.error("Summarization error:", err);
      setSummaries(prev => ({ ...prev, [filename]: `Summarization failed: ${err}` })); // Keep error message in state
    }
  };

   // Placeholder for file removal - needs backend implementation
  // @ts-ignore
  const handleRemoveFile = async (filename: string) => {
     // TODO: Implement backend endpoint for file deletion
     // For now, just remove from frontend state
     try {
         const res = await fetch(`${BACKEND_URL}/delete_file`, {
             method: "POST",
             headers: { "Content-Type": "application/json" },
             body: JSON.stringify({ filename }),
         });
         if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

         // Remove from frontend state only after successful backend deletion
         setUploadedFiles(prev => prev.filter(file => file !== filename));
         setSummaries(prev => { delete prev[filename]; return { ...prev }; });
         if (selectedFiles.includes(filename)) {
             setSelectedFiles(prev => prev.filter(file => file !== filename));
         }
         // Also clear the selected file name in the upload box if the deleted file was selected there
         // This might require passing down a state setter or ref from PDFUpload, 
         // but for now, we assume setSelectedFiles([]) is sufficient if the file was selected in the list.
         // If the file was only selected in the upload box via drag/drop without being in the uploadedFiles list, 
         // we'd need a different mechanism to clear the PDFUpload's internal state.
         // Let's assume for now that files in the upload box are also added to the uploadedFiles list upon successful upload.

         alert(`File deleted successfully: ${filename}`); // Suggestion: Use a better notification system
     } catch (err) {
         console.error("File deletion error:", err);
         alert(`Failed to delete file: ${err}`); // Suggestion: Use a better notification system
     }
   };

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  const handleFileSelect = (filename: string) => {
      // Toggle selection: if already selected, unselect. Otherwise, select just this one.
      if (selectedFiles.includes(filename)) {
          setSelectedFiles([]); // Clear selection
      } else {
          setSelectedFiles([filename]); // Select this file
      }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <Sidebar 
        files={uploadedFiles} 
        onSummarize={handleSummarize} 
        summaries={summaries}
        isCollapsed={isSidebarCollapsed}
        onFileSelect={handleFileSelect}
        selectedFiles={selectedFiles}
        onRemoveFile={handleRemoveFile}
        onToggleCollapse={toggleSidebar} // Pass the toggle function
      />

      {/* Main Content Area */}
      {/* Added some padding here to the main content container */}
      <main className="flex-1 flex flex-col p-8 overflow-hidden bg-white">
         {/* Header with Title and Tagline */}
        <header className="mb-8">
          <h1 className="text-5xl font-extrabold text-blue-900 mb-2">MedPub</h1>
          <p className="text-xl text-gray-600">Your AI-powered assistant for summarizing and exploring medical publications</p>
        </header>

        {/* Content area: Upload, Summary, and Chat in a single column */}
        {/* This outer div manages the vertical stacking and gap */}
        <div className="flex-1 flex flex-col gap-8 overflow-y-auto">
           {/* Upload component always visible */}
           <PDFUpload onUpload={handleUpload} uploading={uploading} />
              
           {/* Container for Summary and Chat, or the Welcome Message */}
           {/* This container needs to take up the remaining space */} 
           {selectedFiles.length > 0 ? (
             <div className="flex-1 flex flex-col gap-8">
                {/* Display summary only if ONE file is selected and summarized */}
                {selectedFiles.length === 1 && summaries[selectedFiles[0]] && summaries[selectedFiles[0]] !== "Summarizing..." && (
                   <SummaryDisplay summary={summaries[selectedFiles[0]]} />
                )}
                 {/* Chat takes remaining vertical space below summary if visible */}
                {/* Ensure chat container grows */} 
                <div className="flex-1 h-full min-h-[400px]">
                   <Chat files={selectedFiles} />{/* Pass selected files to chat */} 
                </div>
             </div>
             ) : (
                // Welcome/Instructional Message when no file is selected
                // This div should also take up remaining space
                <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-600 italic p-8 border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 bg-white">
                    <svg className="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18m-6 4h6m-6 4h6"></path></svg>
                    <p className="text-lg font-semibold mb-2">Welcome to MedPub!</p>
                    <p>Upload a medical paper to get started. You can drag and drop a PDF above or click to browse your files.</p>
                    <p className="mt-4 text-sm">Once uploaded and summarized, you can chat with the document and get AI-powered insights.</p>
                </div>
            )}
        </div>
      </main>
       {/* Suggestion: Add a button here or in the header to toggle the sidebar collapse state */}
    </div>
  );
};

export default App;
