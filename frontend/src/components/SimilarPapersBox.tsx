import React, { useState, useEffect } from 'react';

interface SimilarPaper {
  id: string;
  title: string;
  abstract: string;
  similarity_score: number;
  rank: number;
  arxiv_url: string | null;
}

interface SimilarPapersResponse {
  papers: SimilarPaper[];
  total_found: number;
  filename: string;
  query_length: number;
}

interface SimilarPapersBoxProps {
  filename: string;
  backendUrl: string;
  onError?: (error: string) => void;
}

const SimilarPapersBox: React.FC<SimilarPapersBoxProps> = ({ 
  filename, 
  backendUrl,
  onError 
}) => {
  const [papers, setPapers] = useState<SimilarPaper[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Fetch similar papers when filename changes
  useEffect(() => {
    if (!filename) return;
    
    const fetchSimilarPapers = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`${backendUrl}/similar-papers/${filename}?limit=3`);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data: SimilarPapersResponse = await response.json();
        setPapers(data.papers || []);
        
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch similar papers';
        setError(errorMessage);
        onError?.(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchSimilarPapers();
  }, [filename, backendUrl, onError]);

  // Format filename for display (remove timestamp prefix)
  const getDisplayFilename = (filename: string) => {
    return filename.replace(/^\d{8}_\d{6}_/, '');
  };

  // Truncate abstract to a reasonable length
  const truncateAbstract = (abstract: string, maxLength: number = 200) => {
    if (abstract.length <= maxLength) return abstract;
    return `${abstract.substring(0, maxLength)}...`;
  };

  return (
    <div className="bg-white dark:bg-zinc-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 mb-6">
      {/* Header */}
      <div 
        className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700 cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <svg 
              className="w-5 h-5" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth="2" 
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Similar Research Papers
          </h3>
          {!isCollapsed && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Research similar to: {getDisplayFilename(filename)}
            </p>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {loading && (
            <svg 
              className="animate-spin h-4 w-4 text-blue-600 dark:text-blue-400" 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24"
            >
              <circle 
                className="opacity-25" 
                cx="12" 
                cy="12" 
                r="10" 
                stroke="currentColor" 
                strokeWidth="4"
              />
              <path 
                className="opacity-75" 
                fill="currentColor" 
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l2-2.647z"
              />
            </svg>
          )}
          <button className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
            {isCollapsed ? '+' : '-'}
          </button>
        </div>
      </div>

      {/* Content */}
      {!isCollapsed && (
        <div className="p-6">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <svg 
                  className="animate-spin h-8 w-8 text-blue-600 dark:text-blue-400 mx-auto mb-3" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24"
                >
                  <circle 
                    className="opacity-25" 
                    cx="12" 
                    cy="12" 
                    r="10" 
                    stroke="currentColor" 
                    strokeWidth="4"
                  />
                  <path 
                    className="opacity-75" 
                    fill="currentColor" 
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l2-2.647z"
                  />
                </svg>
                <p className="text-gray-900 dark:text-gray-100">Finding similar papers...</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">This may take a moment</p>
              </div>
            </div>
          ) : error ? (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <svg 
                  className="w-5 h-5 text-red-500 dark:text-red-400" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth="2" 
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                  />
                </svg>
                <h4 className="text-sm font-medium text-red-800 dark:text-red-300">Unable to find similar papers</h4>
              </div>
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              <p className="text-xs text-red-500 dark:text-red-400 mt-2">
                Make sure the arXiv search system is initialized and try again.
              </p>
            </div>
          ) : papers.length === 0 ? (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              <svg 
                className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-3" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth="2" 
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <p>No similar papers found</p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">Try uploading a different document</p>
            </div>
          ) : (
            <div className="space-y-4">
              {papers.map((paper, index) => (
                <div 
                  key={paper.id || index} 
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow duration-200 bg-gray-50 dark:bg-gray-800/50"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center gap-2">
                      <span className="bg-blue-600 dark:bg-blue-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                        #{paper.rank || index + 1}
                      </span>
                      <span className="text-xs text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30 px-2 py-1 rounded-full">
                        {(paper.similarity_score * 100).toFixed(1)}% match
                      </span>
                    </div>
                    {paper.arxiv_url && (
                      <a
                        href={paper.arxiv_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline flex items-center gap-1"
                      >
                        View on arXiv
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                        </svg>
                      </a>
                    )}
                  </div>

                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2 text-sm leading-tight">
                    {paper.title || 'Untitled Paper'}
                  </h4>

                  <div className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed mb-3">
                    {paper.abstract ? (
                      <>
                        {truncateAbstract(paper.abstract)}
                        {paper.abstract.length > 200 && paper.arxiv_url && (
                          <a
                            href={paper.arxiv_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 ml-1 text-xs font-medium underline"
                          >
                            Read more
                          </a>
                        )}
                      </>
                    ) : (
                      <span className="italic text-gray-400 dark:text-gray-500">No abstract available</span>
                    )}
                  </div>

                  {paper.id && (
                    <div className="flex justify-between items-center text-xs text-gray-400 dark:text-gray-500">
                      <span>arXiv ID: {paper.id}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {papers.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
                Showing {papers.length} most similar research papers from arXiv
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SimilarPapersBox; 