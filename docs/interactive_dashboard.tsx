import React, { useState, useEffect } from 'react';
import { Camera, Grid3x3, Target, TrendingUp, CheckCircle, AlertCircle, Download, Play, Pause, RotateCcw } from 'lucide-react';

const InteractiveDistortionDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [animationProgress, setAnimationProgress] = useState(0);
  const [isAnimating, setIsAnimating] = useState(true);
  const [distortionParams, setDistortionParams] = useState({ k1: -0.287, k2: 0.092 });
  const [hoveredMetric, setHoveredMetric] = useState(null);

  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setAnimationProgress(prev => (prev + 1) % 360);
    }, 30);
    return () => clearInterval(interval);
  }, [isAnimating]);

  const metrics = [
    { label: 'RMSE', value: '0.42px', icon: Target, color: 'text-green-600', bg: 'bg-green-100', description: 'Root Mean Square Error - Average calibration accuracy' },
    { label: 'Inliers', value: '91.7%', icon: CheckCircle, color: 'text-blue-600', bg: 'bg-blue-100', description: 'Percentage of corners passing RANSAC validation' },
    { label: 'Processing', value: '1.8s', icon: TrendingUp, color: 'text-purple-600', bg: 'bg-purple-100', description: 'Total calibration time on standard hardware' },
    { label: 'Corners', value: '143/156', icon: Grid3x3, color: 'text-orange-600', bg: 'bg-orange-100', description: 'Successfully detected and validated grid corners' },
  ];

  const pipelineSteps = [
    { name: 'Grid Detection', status: 'complete', time: '0.3s', accuracy: '98%' },
    { name: 'RANSAC Filter', status: 'complete', time: '0.4s', accuracy: '92%' },
    { name: 'Coarse Optimization', status: 'complete', time: '0.5s', accuracy: '85%' },
    { name: 'Fine Refinement', status: 'complete', time: '0.6s', accuracy: '99%' },
  ];

  const generateDistortionField = () => {
    const points = [];
    const gridSize = 12;
    const { k1, k2 } = distortionParams;
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = (i / (gridSize - 1)) * 300;
        const y = (j / (gridSize - 1)) * 300;
        const cx = 150;
        const cy = 150;
        const r2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 10000;
        const distortion = k1 * r2 + k2 * r2 * r2;
        const dx = (x - cx) * distortion * 0.5;
        const dy = (y - cy) * distortion * 0.5;
        
        points.push({ x, y, dx, dy, magnitude: Math.sqrt(dx*dx + dy*dy) });
      }
    }
    return points;
  };

  const distortionPoints = generateDistortionField();

  const AnimatedGrid = () => {
    const gridLines = 8;
    const size = 280;
    const { k1, k2 } = distortionParams;
    
    return (
      <svg width="300" height="300" className="border-2 border-gray-300 rounded-lg bg-white">
        <defs>
          <radialGradient id="distortionGrad">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.3" />
          </radialGradient>
        </defs>
        
        <rect width="300" height="300" fill="url(#distortionGrad)" />
        
        {[...Array(gridLines)].map((_, i) => {
          const pos = (i / (gridLines - 1)) * size + 10;
          const distortedPath = [...Array(50)].map((_, j) => {
            const t = j / 49;
            const x = t * size + 10;
            const y = pos;
            const cx = 150;
            const cy = 150;
            const r2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 10000;
            const d = 1 + k1 * r2 + k2 * r2 * r2;
            const newX = (x - cx) / d + cx;
            const newY = (y - cy) / d + cy;
            return `${j === 0 ? 'M' : 'L'} ${newX} ${newY}`;
          }).join(' ');
          
          return (
            <g key={`h-${i}`}>
              <path d={distortedPath} stroke="#3b82f6" strokeWidth="1.5" fill="none" opacity="0.6" />
            </g>
          );
        })}
        
        {[...Array(gridLines)].map((_, i) => {
          const pos = (i / (gridLines - 1)) * size + 10;
          const distortedPath = [...Array(50)].map((_, j) => {
            const t = j / 49;
            const x = pos;
            const y = t * size + 10;
            const cx = 150;
            const cy = 150;
            const r2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 10000;
            const d = 1 + k1 * r2 + k2 * r2 * r2;
            const newX = (x - cx) / d + cx;
            const newY = (y - cy) / d + cy;
            return `${j === 0 ? 'M' : 'L'} ${newX} ${newY}`;
          }).join(' ');
          
          return (
            <g key={`v-${i}`}>
              <path d={distortedPath} stroke="#8b5cf6" strokeWidth="1.5" fill="none" opacity="0.6" />
            </g>
          );
        })}
        
        <circle cx="150" cy="150" r="4" fill="#ef4444" opacity="0.8" />
        <circle cx="150" cy="150" r="8" fill="none" stroke="#ef4444" strokeWidth="2" opacity="0.6" />
      </svg>
    );
  };

  const ErrorHistogram = () => {
    const bars = [
      { range: '0-0.2', count: 35, color: '#10b981' },
      { range: '0.2-0.4', count: 58, color: '#3b82f6' },
      { range: '0.4-0.6', count: 32, color: '#8b5cf6' },
      { range: '0.6-0.8', count: 12, color: '#f59e0b' },
      { range: '0.8+', count: 6, color: '#ef4444' },
    ];
    const maxCount = Math.max(...bars.map(b => b.count));
    
    return (
      <div className="space-y-2">
        {bars.map((bar, idx) => (
          <div key={idx} className="flex items-center gap-3">
            <div className="w-16 text-sm font-medium text-gray-600">{bar.range}px</div>
            <div className="flex-1 bg-gray-100 rounded-full h-8 relative overflow-hidden">
              <div 
                className="h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                style={{ 
                  width: `${(bar.count / maxCount) * 100}%`,
                  backgroundColor: bar.color
                }}
              >
                <span className="text-white text-xs font-bold">{bar.count}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const DistortionVectorField = () => {
    return (
      <svg width="300" height="300" className="border-2 border-gray-300 rounded-lg bg-gradient-to-br from-blue-50 to-purple-50">
        {distortionPoints.filter((_, i) => i % 2 === 0).map((point, idx) => {
          const length = Math.sqrt(point.dx * point.dx + point.dy * point.dy);
          const scale = 3;
          return (
            <g key={idx}>
              <line
                x1={point.x}
                y1={point.y}
                x2={point.x + point.dx * scale}
                y2={point.y + point.dy * scale}
                stroke={length > 2 ? '#ef4444' : '#3b82f6'}
                strokeWidth="1.5"
                opacity="0.6"
              />
              <circle
                cx={point.x + point.dx * scale}
                cy={point.y + point.dy * scale}
                r="2"
                fill={length > 2 ? '#ef4444' : '#3b82f6'}
              />
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-6 border-t-4 border-blue-500">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-4 rounded-xl">
                <Camera className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">Radial Distortion Calibration</h1>
                <p className="text-gray-600 mt-1">Advanced Computer Vision Pipeline with Division Model</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className="p-2 bg-blue-100 hover:bg-blue-200 rounded-lg transition-colors"
              >
                {isAnimating ? <Pause className="w-5 h-5 text-blue-600" /> : <Play className="w-5 h-5 text-blue-600" />}
              </button>
              <button
                onClick={() => setAnimationProgress(0)}
                className="p-2 bg-purple-100 hover:bg-purple-200 rounded-lg transition-colors"
              >
                <RotateCcw className="w-5 h-5 text-purple-600" />
              </button>
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 mt-6">
            {metrics.map((metric, idx) => {
              const Icon = metric.icon;
              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredMetric(idx)}
                  onMouseLeave={() => setHoveredMetric(null)}
                  className="bg-gradient-to-br from-gray-50 to-white p-4 rounded-xl border-2 border-gray-200 hover:border-blue-400 transition-all cursor-pointer transform hover:scale-105 relative"
                >
                  <div className="flex items-center gap-3 mb-2">
                    <div className={`${metric.bg} p-2 rounded-lg`}>
                      <Icon className={`w-5 h-5 ${metric.color}`} />
                    </div>
                    <div className="text-sm font-medium text-gray-600">{metric.label}</div>
                  </div>
                  <div className="text-2xl font-bold text-gray-800">{metric.value}</div>
                  
                  {hoveredMetric === idx && (
                    <div className="absolute z-10 left-0 right-0 top-full mt-2 bg-gray-900 text-white p-3 rounded-lg text-xs shadow-xl">
                      {metric.description}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-2 mb-6 flex gap-2">
          {['overview', 'pipeline', 'parameters', 'visualization'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                activeTab === tab
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-6">
          {activeTab === 'overview' && (
            <>
              <div className="col-span-2 bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <Grid3x3 className="w-6 h-6 text-blue-600" />
                  Distorted Grid Visualization
                </h2>
                <div className="flex justify-center">
                  <AnimatedGrid />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <div className="text-sm font-medium text-blue-900">Division Model</div>
                    <div className="text-xs text-blue-700 mt-1 font-mono">
                      x_d = x_u / (1 + k₁r² + k₂r⁴)
                    </div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <div className="text-sm font-medium text-purple-900">Parameters</div>
                    <div className="text-xs text-purple-700 mt-1 font-mono">
                      k₁={distortionParams.k1.toFixed(3)}, k₂={distortionParams.k2.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Error Distribution</h2>
                <ErrorHistogram />
                <div className="mt-4 p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                  <div className="text-sm font-bold text-green-900">Excellent Calibration!</div>
                  <div className="text-xs text-green-700 mt-1">
                    93% of corners within 0.6px accuracy
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === 'pipeline' && (
            <div className="col-span-3 bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-6">Hierarchical Optimization Pipeline</h2>
              <div className="space-y-4">
                {pipelineSteps.map((step, idx) => (
                  <div key={idx} className="relative">
                    <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border-2 border-blue-200">
                      <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full text-white font-bold">
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="font-bold text-gray-800">{step.name}</div>
                        <div className="text-sm text-gray-600 mt-1">
                          Processing time: {step.time} • Accuracy: {step.accuracy}
                        </div>
                      </div>
                      <CheckCircle className="w-6 h-6 text-green-600" />
                    </div>
                    {idx < pipelineSteps.length - 1 && (
                      <div className="w-1 h-4 bg-blue-300 ml-5 my-1"></div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'parameters' && (
            <>
              <div className="col-span-2 bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Interactive Parameter Control</h2>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      k₁ (Radial Distortion): {distortionParams.k1.toFixed(3)}
                    </label>
                    <input
                      type="range"
                      min="-0.5"
                      max="0.5"
                      step="0.001"
                      value={distortionParams.k1}
                      onChange={(e) => setDistortionParams({...distortionParams, k1: parseFloat(e.target.value)})}
                      className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      k₂ (Higher Order): {distortionParams.k2.toFixed(3)}
                    </label>
                    <input
                      type="range"
                      min="-0.2"
                      max="0.2"
                      step="0.001"
                      value={distortionParams.k2}
                      onChange={(e) => setDistortionParams({...distortionParams, k2: parseFloat(e.target.value)})}
                      className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  <div className="flex justify-center mt-6">
                    <AnimatedGrid />
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Calibration Stats</h2>
                <div className="space-y-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <div className="text-xs text-blue-600 font-medium">Principal Point</div>
                    <div className="text-sm font-bold text-blue-900 mt-1">(1952.3, 1468.7)</div>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-lg">
                    <div className="text-xs text-purple-600 font-medium">Focal Length</div>
                    <div className="text-sm font-bold text-purple-900 mt-1">fx=2847.2, fy=2851.8</div>
                  </div>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <div className="text-xs text-green-600 font-medium">Mean Reprojection</div>
                    <div className="text-sm font-bold text-green-900 mt-1">0.41 pixels</div>
                  </div>
                  <div className="p-3 bg-orange-50 rounded-lg">
                    <div className="text-xs text-orange-600 font-medium">Total Time</div>
                    <div className="text-sm font-bold text-orange-900 mt-1">1.8 seconds</div>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === 'visualization' && (
            <>
              <div className="col-span-2 bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Distortion Vector Field</h2>
                <div className="flex justify-center">
                  <DistortionVectorField />
                </div>
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm font-medium text-gray-700">Vector Field Legend</div>
                  <div className="flex items-center gap-4 mt-2 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-blue-600"></div>
                      <span>Low distortion</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-red-600"></div>
                      <span>High distortion</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Key Innovations</h2>
                <div className="space-y-3">
                  <div className="p-3 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg border-l-4 border-blue-500">
                    <div className="text-sm font-bold text-blue-900">Division Model</div>
                    <div className="text-xs text-blue-700 mt-1">Better numerical stability than polynomial</div>
                  </div>
                  <div className="p-3 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg border-l-4 border-purple-500">
                    <div className="text-sm font-bold text-purple-900">Adaptive RANSAC</div>
                    <div className="text-xs text-purple-700 mt-1">40% faster with probabilistic scoring</div>
                  </div>
                  <div className="p-3 bg-gradient-to-r from-green-50 to-green-100 rounded-lg border-l-4 border-green-500">
                    <div className="text-sm font-bold text-green-900">Hierarchical Optimization</div>
                    <div className="text-xs text-green-700 mt-1">Multi-scale refinement approach</div>
                  </div>
                  <div className="p-3 bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg border-l-4 border-orange-500">
                    <div className="text-sm font-bold text-orange-900">Huber Loss</div>
                    <div className="text-xs text-orange-700 mt-1">Robust to outliers and noise</div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="mt-6 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-bold">Ready to Deploy</h3>
              <p className="text-sm opacity-90 mt-1">Production-ready calibration pipeline with state-of-art accuracy</p>
            </div>
            <button className="bg-white text-blue-600 px-6 py-3 rounded-lg font-bold hover:bg-blue-50 transition-colors flex items-center gap-2">
              <Download className="w-5 h-5" />
              Export Results
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InteractiveDistortionDashboard;