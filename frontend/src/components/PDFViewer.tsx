import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';

// Set up the worker for react-pdf with updated configuration for version compatibility
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url,
).toString();

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

const LoadingSpinner = () => (
  <div className="flex flex-col items-center justify-center h-full space-y-4">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
    <div className="text-gray-600 font-medium">Loading PDF...</div>
    <div className="text-sm text-gray-500">Please wait while we load the document</div>
  </div>
);

const PDFViewer: React.FC<PDFViewerProps> = ({ filename, backendUrl }) => {
  const [sections, setSections] = useState<DocumentSection[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState<number>(0);
  const [scale, setScale] = useState(1.0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [documentReady, setDocumentReady] = useState(false);

  // Construct the full PDF URL with proper path
  const pdfUrl = `${backendUrl}/uploads/${filename}`;

  // Improved PDF loading options
  const pdfOptions = {
    cMapUrl: 'https://unpkg.com/pdfjs-dist@4.4.168/cmaps/',
    cMapPacked: true,
    standardFontDataUrl: 'https://unpkg.com/pdfjs-dist@4.4.168/standard_fonts/',
    isEvalSupported: false,
    isOffscreenCanvasSupported: false,
  };

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

    if (filename) {
      fetchDocumentInfo();
    }
  }, [filename, backendUrl]);

  // Navigate to section
  const navigateToSection = (sectionPage: number) => {
    setCurrentPage(sectionPage);
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    console.log('PDF loaded successfully with', numPages, 'pages');
    setTotalPages(numPages);
    setLoading(false);
    setError(null);
    setDocumentReady(true);
  };

  const onDocumentLoadError = (error: Error) => {
    console.error('PDF loading error:', error);
    setError(`Failed to load PDF: ${error.message}`);
    setLoading(false);
    setDocumentReady(false);
  };

  const onPageLoadSuccess = () => {
    console.log('Page loaded successfully:', currentPage);
  };

  const onPageLoadError = (error: Error) => {
    console.error('Page loading error:', error);
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
          <p className="text-xs text-gray-600">Status: {documentReady ? 'Ready' : 'Loading'}</p>
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
                disabled={currentPage <= 1 || loading}
                className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="text-sm text-gray-600">
                {currentPage} / {totalPages || '?'}
              </span>
              <button
                onClick={goToNextPage}
                disabled={currentPage >= totalPages || loading}
                className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={zoomOut}
              disabled={loading}
              className="px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50"
            >
              -
            </button>
            <button
              onClick={resetZoom}
              disabled={loading}
              className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm disabled:opacity-50"
            >
              {Math.round(scale * 100)}%
            </button>
            <button
              onClick={zoomIn}
              disabled={loading}
              className="px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50"
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
          {loading && <LoadingSpinner />}
          
          {error && (
            <div className="flex items-center justify-center h-full">
              <div className="text-red-500 text-center max-w-md">
                <div className="mb-2 text-4xl">⚠️</div>
                <div className="mb-2 font-semibold">Error loading PDF</div>
                <div className="text-sm mb-4 text-gray-600">{error}</div>
                <div className="space-y-2">
                  <button
                    onClick={() => {
                      setError(null);
                      setLoading(true);
                      setDocumentReady(false);
                    }}
                    className="block w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                  >
                    Try Again
                  </button>
                  <a
                    href={pdfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block w-full px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                  >
                    Open in Browser
                  </a>
                </div>
              </div>
            </div>
          )}
          
          {!error && (
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<LoadingSpinner />}
              error={
                <div className="text-red-500 text-center">
                  <div className="mb-2">❌ Failed to load PDF</div>
                  <div className="text-sm">Please check if the file exists and try again</div>
                </div>
              }
              options={pdfOptions}
            >
              {documentReady && (
                <Page
                  pageNumber={currentPage}
                  scale={scale}
                  onLoadSuccess={onPageLoadSuccess}
                  onLoadError={onPageLoadError}
                  loading={
                    <div className="flex items-center justify-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-3"></div>
                      <span className="text-gray-600">Loading page {currentPage}...</span>
                    </div>
                  }
                  error={
                    <div className="text-red-500 text-center py-8">
                      <div className="mb-2">❌ Failed to load page {currentPage}</div>
                      <div className="text-sm">Please try refreshing or check your connection</div>
                    </div>
                  }
                  className="shadow-lg"
                  renderTextLayer={true}
                  renderAnnotationLayer={false}
                />
              )}
            </Document>
          )}
        </div>
      </div>
    </div>
  );
};

export default PDFViewer; 