import React, { useState, useEffect } from 'react';

interface PDFViewerProps {
  filename: string;
  backendUrl: string;
  onTextSelect?: (text: string, context: string) => void;
  highlightedSection?: { page: number; text: string };
}

interface DocumentSection {
  name: string;
  page: number;
}

const PDFViewer: React.FC<PDFViewerProps> = ({ filename, backendUrl }) => {
  const [sections, setSections] = useState<DocumentSection[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState<number>(0);

  // Fetch document info including sections
  useEffect(() => {
    const fetchDocumentInfo = async () => {
      try {
        const response = await fetch(`${backendUrl}/document-info/${filename}`);
        if (response.ok) {
          const data = await response.json();
          setTotalPages(data.metadata.page_count || 10);
          
          // Map sections to approximate page numbers
          const sectionList = data.sections.map((section: string, index: number) => ({
            name: section,
            page: Math.floor((index / data.sections.length) * (data.metadata.page_count || 10)) + 1
          }));
          setSections(sectionList);
        }
      } catch (err) {
        console.error('Error fetching document info:', err);
      }
    };

    fetchDocumentInfo();
  }, [filename, backendUrl]);

  // Navigate to section
  const navigateToSection = (sectionPage: number) => {
    setCurrentPage(sectionPage);
    // Scroll to page in iframe using hash
    const iframe = document.getElementById('pdf-iframe') as HTMLIFrameElement;
    if (iframe) {
      iframe.src = `${backendUrl}/uploads/${filename}#page=${sectionPage}`;
    }
  };

  const pdfUrl = `${backendUrl}/uploads/${filename}`;

  return (
    <div className="flex h-full">
      {/* Section Navigation Sidebar */}
      <div className="w-64 bg-gray-100 p-4 overflow-y-auto border-r border-gray-300">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Sections</h3>
        <ul className="space-y-2">
          {sections.map((section, index) => (
            <li key={index}>
              <button
                onClick={() => navigateToSection(section.page)}
                className={`w-full text-left px-3 py-2 rounded-md transition-colors duration-200 ${
                  currentPage === section.page
                    ? 'bg-blue-500 text-white'
                    : 'hover:bg-gray-200 text-gray-700'
                }`}
              >
                {section.name}
              </button>
            </li>
          ))}
        </ul>
        
        {/* Document Info */}
        <div className="mt-6 pt-6 border-t border-gray-300">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Document Info</h4>
          <p className="text-xs text-gray-600">Total Pages: {totalPages}</p>
          <p className="text-xs text-gray-600">Current Page: {currentPage}</p>
        </div>
      </div>

      {/* PDF Viewer */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Toolbar */}
        <div className="flex items-center justify-between p-4 bg-white border-b border-gray-300">
          <div className="flex items-center space-x-4">
            <span className="text-gray-700 font-medium">
              PDF Viewer - {filename.replace(/^\d{8}_\d{6}_/, '')}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
            >
              Open in New Tab
            </a>
          </div>
        </div>

        {/* PDF Display */}
        <div className="flex-1 overflow-hidden">
          <iframe
            id="pdf-iframe"
            src={pdfUrl}
            className="w-full h-full"
            title={`PDF Viewer - ${filename}`}
          />
        </div>
      </div>
    </div>
  );
};

export default PDFViewer; 