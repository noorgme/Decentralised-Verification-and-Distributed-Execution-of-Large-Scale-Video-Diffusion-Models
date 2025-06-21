import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import Layout from '../../components/Layout';

const fetcher = (url: string) => fetch(url).then(r => r.json());

// Custom hook to replace useSWR
function usePollingData(url: string | null, interval: number = 3000) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!url) {
      setData(null);
      setLoading(false);
      return;
    }

    const fetchData = async () => {
      try {
        console.log(`🔍 Fetching data from: ${url}`);
        const response = await fetch(url);
        console.log(`📡 Response status: ${response.status}`);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log(`📦 Received data:`, result);
        console.log(`📦 Data type: ${typeof result}`);
        console.log(`📦 Data keys: ${Object.keys(result)}`);
        
        setData(result);
        setError(null);
      } catch (err) {
        console.error(`❌ Error fetching data from ${url}:`, err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [url, interval]);

  return { data, error, loading };
}

export default function StatusPage() {
  const { query } = useRouter();
  const { data: statusData, loading: statusLoading } = usePollingData(
    query.requestId ? `/status/${query.requestId}` : null, 
    3000
  );
  const { data: resultData, loading: resultLoading } = usePollingData(
    statusData?.status === 'completed' ? `/result/${query.requestId}` : null, 
    5000
  );

  const isLoading = statusLoading;
  const isCompleted = statusData?.status === 'completed';
  const hasResults = resultData?.status === 'completed';

  return (
    <Layout>
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="mb-4">
            <h3 className="text-2xl font-bold text-blue-700">Job Status: {query.requestId}</h3>
          </div>
          <div>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600">Loading...</span>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <span className="text-lg font-semibold">Status:</span>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    statusData?.status === 'completed' 
                      ? 'bg-green-100 text-green-800' 
                      : statusData?.status === 'processing'
                      ? 'bg-blue-100 text-blue-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {statusData?.status || 'Unknown'}
                  </span>
                </div>
                
                {statusData?.details?.prompt && (
                  <div>
                    <span className="text-lg font-semibold">Prompt:</span>
                    <p className="mt-1 text-gray-700 bg-gray-50 p-3 rounded-lg">
                      {statusData.details.prompt}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        {isCompleted && hasResults && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <div className="mb-4">
              <h4 className="text-xl font-bold text-green-700">Generated Videos</h4>
            </div>
            <div>
              {/* Summary Statistics */}
              {resultData?.result?.miners && resultData.result.miners.length > 0 && (
                <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                  <h5 className="text-lg font-semibold text-blue-800 mb-3">Summary</h5>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {resultData?.result?.miners?.filter((m: any) => m.status === 'success').length || 0}
                      </div>
                      <div className="text-sm text-gray-600">Successful Videos</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).length > 0 
                          ? Math.max(...resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).map((m: any) => m.score * 100) || []).toFixed(1)
                          : 'N/A'
                        }%
                      </div>
                      <div className="text-sm text-gray-600">Best CLIP Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).length > 0
                          ? (resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).reduce((acc: number, m: any) => acc + m.score, 0) / resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).length * 100).toFixed(1)
                          : 'N/A'
                        }%
                      </div>
                      <div className="text-sm text-gray-600">Average CLIP Score</div>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {resultData?.result?.miners?.map((miner: any, index: number) => (
                  <div
                    key={miner.uid}
                    className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200"
                  >
                    <div className="p-4">
                      <div className="flex justify-between items-center mb-3">
                        <h5 className="font-semibold text-gray-800">Miner {miner.uid}</h5>
                        <div className="flex items-center space-x-2">
                          {resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).length > 0 && 
                           miner.status === 'success' && 
                           miner.score === Math.max(...resultData?.result?.miners?.filter((m: any) => m.status === 'success' && m.score !== undefined).map((m: any) => m.score) || []) && (
                            <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded-full">
                              🏆 Best
                            </span>
                          )}
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            miner.status === 'success' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {miner.status}
                          </span>
                        </div>
                      </div>
                      
                      {miner.status === 'success' && miner.video_url && (
                        <div className="space-y-3">
                          {(() => {
                            console.log(`🎥 Rendering video for miner ${miner.uid}:`);
                            console.log(`   - video_url: ${miner.video_url}`);
                            console.log(`   - video_path: ${miner.video_path}`);
                            console.log(`   - status: ${miner.status}`);
                            console.log(`   - score: ${miner.score}`);
                            
                            const videoSrc = `http://localhost:8080${miner.video_url}`;
                            console.log(`   - Full video src: ${videoSrc}`);
                            
                            return (
                              <video
                                className="w-full h-48 object-cover rounded-lg bg-gray-100"
                                controls
                                preload="metadata"
                                poster="/video-placeholder.svg"
                                onLoadStart={() => console.log(`🎬 Video loading started for miner ${miner.uid}`)}
                                onCanPlay={() => console.log(`✅ Video can play for miner ${miner.uid}`)}
                                onError={(e) => console.error(`❌ Video error for miner ${miner.uid}:`, e)}
                              >
                                <source src={videoSrc} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                            );
                          })()}
                          
                          {/* CLIP Score Display */}
                          {miner.score !== undefined && (
                            <div className="space-y-2">
                              <div className="text-center">
                                <div className="flex items-center justify-center space-x-1 mb-1">
                                  <span className="text-sm font-medium text-gray-600">CLIP Quality Score</span>
                                  <div className="group relative">
                                    <span className="text-gray-400 cursor-help">ℹ️</span>
                                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap z-10">
                                      Measures how well the video matches the text prompt using CLIP similarity
                                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center justify-center space-x-2 mt-1">
                                  <div className="text-2xl font-bold text-blue-600">
                                    {(miner.score * 100).toFixed(1)}%
                                  </div>
                                  <div className="flex items-center">
                                    {miner.score >= 0.8 ? (
                                      <span className="text-green-500">⭐</span>
                                    ) : miner.score >= 0.6 ? (
                                      <span className="text-yellow-500">⭐</span>
                                    ) : (
                                      <span className="text-red-500">⭐</span>
                                    )}
                                  </div>
                                </div>
                              </div>
                              
                              {/* Score Bar */}
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full transition-all duration-300 ${
                                    miner.score >= 0.8 
                                      ? 'bg-green-500' 
                                      : miner.score >= 0.6 
                                      ? 'bg-yellow-500' 
                                      : 'bg-red-500'
                                  }`}
                                  style={{ width: `${miner.score * 100}%` }}
                                ></div>
                              </div>
                              
                              {/* Score Description */}
                              <div className="text-center">
                                <span className={`text-xs font-medium ${
                                  miner.score >= 0.8 
                                    ? 'text-green-600' 
                                    : miner.score >= 0.6 
                                    ? 'text-yellow-600' 
                                    : 'text-red-600'
                                }`}>
                                  {miner.score >= 0.8 
                                    ? 'Excellent - High prompt fidelity' 
                                    : miner.score >= 0.6 
                                    ? 'Good - Moderate prompt fidelity' 
                                    : 'Poor - Low prompt fidelity'
                                  }
                                </span>
                              </div>
                              
                              {/* Technical Details */}
                              <div className="bg-gray-50 rounded-lg p-2 text-xs text-gray-600">
                                <div className="grid grid-cols-2 gap-1">
                                  <div>Raw Score:</div>
                                  <div className="font-mono">{miner.score.toFixed(4)}</div>
                                  <div>Range:</div>
                                  <div>0.0 - 1.0</div>
                                  <div>Metric:</div>
                                  <div>CLIP Similarity</div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {miner.status !== 'success' && (
                        <div className="text-center py-8 text-gray-500">
                          <div className="text-red-500 mb-2">
                            {miner.error || 'Generation failed'}
                          </div>
                          <div className="text-sm">
                            Status: {miner.status}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              {resultData?.result?.miners?.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No videos were generated successfully.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Processing Status */}
        {isCompleted && !hasResults && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <div className="mb-4">
              <h4 className="text-xl font-bold text-blue-700">Processing Results</h4>
            </div>
            <div>
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600">Processing results...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
} 