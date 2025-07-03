import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';

// Set up the worker for react-pdf
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

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
  const [scale, setScale] = useState(1.0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Construct the full PDF URL with proper path
  const pdfUrl = `${backendUrl}/uploads/${filename}`;

  // Fetch document info including sections
  useEffect(() => {
    const fetchDocumentInfo = async () => {
      try {
        const response = await fetch(`${backendUrl}/document-info/${filename}`);
        if (response.ok) {
          const data = await response.json();
          setTotalPages(data.metadata.page_count || 1);
          
          // Map sections to approximate page numbers
          const sectionList = data.sections.map((section: string, index: number) => ({
            name: section,
            page: Math.floor((index / data.sections.length) * (data.metadata.page_count || 1)) + 1
          }));
          setSections(sectionList);
        }
      } catch (err) {
        console.error('Error fetching document info:', err);
        // Continue anyway, PDF might still load
      }
    };

    fetchDocumentInfo();
  }, [filename, backendUrl]);

  // Navigate to section
  const navigateToSection = (sectionPage: number) => {
    setCurrentPage(sectionPage);
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setTotalPages(numPages);
    setLoading(false);
    setError(null);
  };

  const onDocumentLoadError = (error: Error) => {
    setError(`Failed to load PDF: ${error.message}`);
    setLoading(false);
  };

  const goToPrevPage = () => {
    setCurrentPage(prev => Math.max(1, prev - 1));
  };

  const goToNextPage = () => {
    setCurrentPage(prev => Math.min(totalPages, prev + 1));
  };

  const zoomIn = () => {
    setScale(prev => Math.min(2.0, prev + 0.2));
  };

  const zoomOut = () => {
    setScale(prev => Math.max(0.5, prev - 0.2));
  };

  const resetZoom = () => {
    setScale(1.0);
  };

  return (
    <div className="flex h-full bg-gray-100">
      {/* Section Navigation Sidebar */}
      <div className="w-64 bg-white p-4 overflow-y-auto border-r border-gray-300 shadow-sm">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Sections</h3>
        <ul className="space-y-2">
          {sections.map((section, index) => (
            <li key={index}>
              <button
                onClick={() => navigateToSection(section.page)}
                className={`w-full text-left px-3 py-2 rounded-md transition-colors duration-200 text-sm ${
                  currentPage === section.page
                    ? 'bg-blue-500 text-white'
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <div className="font-medium truncate">{section.name}</div>
                <div className="text-xs opacity-75">Page {section.page}</div>
              </button>
            </li>
          ))}
        </ul>
        
        {/* Document Info */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Document Info</h4>
          <p className="text-xs text-gray-600">Total Pages: {totalPages}</p>
          <p className="text-xs text-gray-600">Current Page: {currentPage}</p>
          <p className="text-xs text-gray-600">Zoom: {Math.round(scale * 100)}%</p>
        </div>
      </div>

      {/* PDF Viewer */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Toolbar */}
        <div className="flex items-center justify-between p-4 bg-white border-b border-gray-200 shadow-sm">
          <div className="flex items-center space-x-4">
            <span className="text-gray-700 font-medium text-sm">
              {filename.replace(/^\d{8}_\d{6}_/, '')}
            </span>
            <div className="flex items-center space-x-2">
              <button
                onClick={goToPrevPage}
                disabled={currentPage <= 1}
                className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="text-sm text-gray-600">
                {currentPage} / {totalPages}
              </span>
              <button
                onClick={goToNextPage}
                disabled={currentPage >= totalPages}
                className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={zoomOut}
              className="px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              -
            </button>
            <button
              onClick={resetZoom}
              className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
            >
              {Math.round(scale * 100)}%
            </button>
            <button
              onClick={zoomIn}
              className="px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              +
            </button>
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
            >
              Open in New Tab
            </a>
          </div>
        </div>

        {/* PDF Display */}
        <div className="flex-1 overflow-auto flex justify-center py-4">
          {loading && (
            <div className="flex items-center justify-center h-full">
              <div className="text-gray-500">Loading PDF...</div>
            </div>
          )}
          
          {error && (
            <div className="flex items-center justify-center h-full">
              <div className="text-red-500 text-center">
                <div className="mb-2">⚠️ Error loading PDF</div>
                <div className="text-sm">{error}</div>
                <a
                  href={pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-4 inline-block px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Open in Browser
                </a>
              </div>
            </div>
          )}
          
          {!loading && !error && (
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<div className="text-gray-500">Loading document...</div>}
              error={<div className="text-red-500">Failed to load PDF</div>}
            >
              <Page
                pageNumber={currentPage}
                scale={scale}
                loading={<div className="text-gray-500">Loading page...</div>}
                error={<div className="text-red-500">Failed to load page</div>}
                className="shadow-lg"
              />
            </Document>
          )}
        </div>
      </div>
    </div>
  );
};

export default PDFViewer; 