import React, { useMemo, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitBranch,
  ArrowDown,
  ArrowRight,
  Filter,
  Target,
  Users,
  BarChart3,
  Info,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Download,
  Search
} from 'lucide-react';
import { cn } from '../../utils';

interface DecisionNode {
  id: string;
  type: 'decision' | 'leaf';
  feature?: string;
  threshold?: number;
  operator?: '>' | '<=' | '==' | '!=';
  value?: number | string;
  samples: number;
  purity: number;
  prediction?: number | string;
  confidence?: number;
  children?: DecisionNode[];
  depth: number;
  path?: string[];
  gini?: number;
  entropy?: number;
  classDistribution?: { [key: string]: number };
}

interface DecisionTreeChartProps {
  treeData: DecisionNode;
  maxDepth?: number;
  modelType: 'classification' | 'regression';
  className?: string;
  interactive?: boolean;
  onNodeClick?: (node: DecisionNode) => void;
  highlightPath?: string[];
  showMetrics?: boolean;
}

const DecisionTreeChart: React.FC<DecisionTreeChartProps> = ({
  treeData,
  maxDepth = 5,
  modelType,
  className,
  interactive = true,
  onNodeClick,
  highlightPath = [],
  showMetrics = true
}) => {
  const [selectedNode, setSelectedNode] = useState<DecisionNode | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [expandedDepth, setExpandedDepth] = useState(maxDepth);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Constants for tree layout
  const NODE_WIDTH = 200;
  const NODE_HEIGHT = 80;
  const LEVEL_HEIGHT = 120;
  const HORIZONTAL_SPACING = 50;

  // Calculate tree layout
  const treeLayout = useMemo(() => {
    const calculateLayout = (node: DecisionNode, depth: number, x: number, parentX?: number): any => {
      const nodeData = {
        ...node,
        x,
        y: depth * LEVEL_HEIGHT + 50,
        depth
      };

      if (!node.children || depth >= expandedDepth) {
        return {
          node: nodeData,
          width: NODE_WIDTH,
          children: []
        };
      }

      const childLayouts = node.children.map((child, index) => {
        const childX = x + (index - (node.children!.length - 1) / 2) * (NODE_WIDTH + HORIZONTAL_SPACING);
        return calculateLayout(child, depth + 1, childX, x);
      });

      const totalWidth = childLayouts.reduce((sum, child) => sum + child.width, 0) + 
                        (childLayouts.length - 1) * HORIZONTAL_SPACING;

      return {
        node: nodeData,
        width: Math.max(NODE_WIDTH, totalWidth),
        children: childLayouts
      };
    };

    return calculateLayout(treeData, 0, 0);
  }, [treeData, expandedDepth]);

  // Flatten tree for searching and highlighting
  const flattenedNodes = useMemo(() => {
    const flatten = (layout: any): any[] => {
      return [layout.node, ...layout.children.flatMap(flatten)];
    };
    return flatten(treeLayout);
  }, [treeLayout]);

  // Filter nodes based on search
  const filteredNodes = useMemo(() => {
    if (!searchTerm) return flattenedNodes;
    
    return flattenedNodes.filter(node => 
      node.feature?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (node.prediction && String(node.prediction).toLowerCase().includes(searchTerm.toLowerCase()))
    );
  }, [flattenedNodes, searchTerm]);

  const handleNodeClick = (node: DecisionNode) => {
    if (interactive) {
      setSelectedNode(node);
      onNodeClick?.(node);
    }
  };

  const getNodeColor = (node: DecisionNode): string => {
    if (highlightPath.includes(node.id)) {
      return '#3b82f6'; // Blue for highlighted path
    }
    
    if (node.type === 'leaf') {
      if (modelType === 'classification') {
        return node.purity > 0.8 ? '#10b981' : node.purity > 0.6 ? '#f59e0b' : '#ef4444';
      } else {
        return '#10b981'; // Green for regression leaves
      }
    }
    
    return '#6b7280'; // Gray for decision nodes
  };

  const getNodeBorderColor = (node: DecisionNode): string => {
    if (selectedNode?.id === node.id) return '#1d4ed8';
    if (hoveredNode === node.id) return '#3b82f6';
    return 'transparent';
  };

  const renderNode = (layout: any) => {
    const { node } = layout;
    const isHighlighted = highlightPath.includes(node.id);
    const isSelected = selectedNode?.id === node.id;
    const isHovered = hoveredNode === node.id;

    return (
      <g key={node.id}>
        {/* Connections to children */}
        {layout.children.map((childLayout: any, index: number) => (
          <g key={`${node.id}-${childLayout.node.id}`}>
            {/* Connection line */}
            <motion.line
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.5, delay: node.depth * 0.1 }}
              x1={node.x}
              y1={node.y + NODE_HEIGHT / 2}
              x2={childLayout.node.x}
              y2={childLayout.node.y - NODE_HEIGHT / 2}
              stroke={isHighlighted ? '#3b82f6' : '#d1d5db'}
              strokeWidth={isHighlighted ? 2 : 1}
              markerEnd="url(#arrowhead)"
            />
            
            {/* Decision label */}
            <text
              x={(node.x + childLayout.node.x) / 2}
              y={(node.y + childLayout.node.y) / 2}
              fill="#6b7280"
              fontSize="12"
              textAnchor="middle"
              className="pointer-events-none"
            >
              {node.children && index === 0 ? 'No' : 'Yes'}
            </text>
          </g>
        ))}

        {/* Node */}
        <motion.g
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3, delay: node.depth * 0.1 }}
          onClick={() => handleNodeClick(node)}
          onMouseEnter={() => setHoveredNode(node.id)}
          onMouseLeave={() => setHoveredNode(null)}
          style={{ cursor: interactive ? 'pointer' : 'default' }}
        >
          {/* Node background */}
          <rect
            x={node.x - NODE_WIDTH / 2}
            y={node.y - NODE_HEIGHT / 2}
            width={NODE_WIDTH}
            height={NODE_HEIGHT}
            fill={getNodeColor(node)}
            fillOpacity={node.type === 'leaf' ? 0.8 : 0.1}
            stroke={getNodeBorderColor(node)}
            strokeWidth={isSelected ? 3 : isHovered ? 2 : 1}
            rx={8}
            className="transition-all duration-200"
          />

          {/* Node icon */}
          <foreignObject
            x={node.x - 12}
            y={node.y - NODE_HEIGHT / 2 + 8}
            width="24"
            height="24"
          >
            {node.type === 'decision' ? (
              <GitBranch className="w-6 h-6 text-gray-600" />
            ) : (
              <Target className="w-6 h-6 text-green-600" />
            )}
          </foreignObject>

          {/* Node text */}
          <text
            x={node.x}
            y={node.y - 15}
            fill="#1f2937"
            fontSize="12"
            fontWeight="bold"
            textAnchor="middle"
            className="pointer-events-none"
          >
            {node.type === 'decision' && node.feature ? 
              `${node.feature} ${node.operator} ${node.threshold}` :
              `Prediction: ${node.prediction}`
            }
          </text>

          {/* Samples count */}
          <text
            x={node.x}
            y={node.y}
            fill="#6b7280"
            fontSize="10"
            textAnchor="middle"
            className="pointer-events-none"
          >
            Samples: {node.samples}
          </text>

          {/* Purity/Confidence */}
          {showMetrics && (
            <text
              x={node.x}
              y={node.y + 15}
              fill="#6b7280"
              fontSize="10"
              textAnchor="middle"
              className="pointer-events-none"
            >
              {node.type === 'leaf' ? 
                `Confidence: ${((node.confidence || 0) * 100).toFixed(1)}%` :
                `Purity: ${(node.purity * 100).toFixed(1)}%`
              }
            </text>
          )}
        </motion.g>

        {/* Render children */}
        {layout.children.map((childLayout: any) => renderNode(childLayout))}
      </g>
    );
  };

  const calculateSVGDimensions = () => {
    const allNodes = flattenedNodes;
    const minX = Math.min(...allNodes.map(n => n.x - NODE_WIDTH / 2));
    const maxX = Math.max(...allNodes.map(n => n.x + NODE_WIDTH / 2));
    const minY = Math.min(...allNodes.map(n => n.y - NODE_HEIGHT / 2));
    const maxY = Math.max(...allNodes.map(n => n.y + NODE_HEIGHT / 2));
    
    return {
      width: (maxX - minX + 100) * zoomLevel,
      height: (maxY - minY + 100) * zoomLevel,
      viewBox: `${minX - 50} ${minY - 50} ${maxX - minX + 100} ${maxY - minY + 100}`
    };
  };

  const { width, height, viewBox } = calculateSVGDimensions();

  const exportTree = () => {
    if (!svgRef.current) return;
    
    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `decision-tree-${Date.now()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-700 shadow-sm',
        isFullscreen ? 'fixed inset-4 z-50' : '',
        className
      )}
    >
      {/* Header */}
      <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
              <GitBranch className="w-5 h-5 text-purple-500" />
              Decision Tree Visualization
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
              Interactive exploration of model decision paths and logic
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={exportTree}
              className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
              title="Export tree as SVG"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Search Nodes
            </label>
            <div className="relative">
              <Search className="absolute left-2 top-2 w-4 h-4 text-neutral-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Feature name or value..."
                className="pl-8 w-full px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Max Depth
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={expandedDepth}
              onChange={(e) => setExpandedDepth(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
              {expandedDepth} levels
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Zoom Level
            </label>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.1))}
                className="p-1 text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300 min-w-16 text-center">
                {(zoomLevel * 100).toFixed(0)}%
              </span>
              <button
                onClick={() => setZoomLevel(Math.min(2, zoomLevel + 0.1))}
                className="p-1 text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Options
            </label>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setZoomLevel(1)}
                className="px-3 py-1 text-xs bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded hover:bg-neutral-200 dark:hover:bg-neutral-700"
              >
                <RotateCcw className="w-3 h-3 mr-1 inline" />
                Reset
              </button>
            </div>
          </div>
        </div>

        {/* Tree Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
            <div className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
              Total Nodes
            </div>
            <div className="text-xl font-bold text-blue-900 dark:text-blue-100">
              {flattenedNodes.length}
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
            <div className="text-xs font-medium text-green-600 dark:text-green-400 uppercase tracking-wider">
              Leaf Nodes
            </div>
            <div className="text-xl font-bold text-green-900 dark:text-green-100">
              {flattenedNodes.filter(n => n.type === 'leaf').length}
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
            <div className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wider">
              Max Depth
            </div>
            <div className="text-xl font-bold text-purple-900 dark:text-purple-100">
              {Math.max(...flattenedNodes.map(n => n.depth))}
            </div>
          </div>
          
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
            <div className="text-xs font-medium text-orange-600 dark:text-orange-400 uppercase tracking-wider">
              Avg Purity
            </div>
            <div className="text-xl font-bold text-orange-900 dark:text-orange-100">
              {(flattenedNodes.filter(n => n.type === 'leaf').reduce((sum, n) => sum + n.purity, 0) / 
                flattenedNodes.filter(n => n.type === 'leaf').length * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>

      {/* Tree Visualization */}
      <div 
        ref={containerRef}
        className="p-6 overflow-auto"
        style={{ height: isFullscreen ? 'calc(100vh - 300px)' : '600px' }}
      >
        <svg
          ref={svgRef}
          width={width}
          height={height}
          viewBox={viewBox}
          className="border border-neutral-200 dark:border-neutral-700 rounded-lg bg-neutral-50 dark:bg-neutral-800"
        >
          {/* Arrow marker definition */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
              fill="#d1d5db"
            >
              <polygon points="0 0, 10 3.5, 0 7" />
            </marker>
          </defs>
          
          {renderNode(treeLayout)}
        </svg>
      </div>

      {/* Node Details Panel */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-neutral-200 dark:border-neutral-700 p-6 bg-neutral-50 dark:bg-neutral-800"
          >
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                Node Details: {selectedNode.id}
              </h4>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              >
                ×
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Type:</span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100 capitalize">
                  {selectedNode.type}
                </span>
              </div>
              
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Samples:</span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  {selectedNode.samples}
                </span>
              </div>
              
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Depth:</span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  {selectedNode.depth}
                </span>
              </div>
              
              {selectedNode.type === 'decision' && selectedNode.feature && (
                <>
                  <div>
                    <span className="font-medium text-neutral-700 dark:text-neutral-300">Feature:</span>
                    <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                      {selectedNode.feature}
                    </span>
                  </div>
                  
                  <div>
                    <span className="font-medium text-neutral-700 dark:text-neutral-300">Threshold:</span>
                    <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                      {selectedNode.operator} {selectedNode.threshold}
                    </span>
                  </div>
                </>
              )}
              
              {selectedNode.type === 'leaf' && (
                <>
                  <div>
                    <span className="font-medium text-neutral-700 dark:text-neutral-300">Prediction:</span>
                    <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                      {selectedNode.prediction}
                    </span>
                  </div>
                  
                  <div>
                    <span className="font-medium text-neutral-700 dark:text-neutral-300">Confidence:</span>
                    <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                      {((selectedNode.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </>
              )}
              
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Purity:</span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  {(selectedNode.purity * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            
            {selectedNode.classDistribution && (
              <div className="mt-4">
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Class Distribution:</span>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-2">
                  {Object.entries(selectedNode.classDistribution).map(([className, count]) => (
                    <div key={className} className="bg-white dark:bg-neutral-700 rounded p-2 text-center">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">{className}</div>
                      <div className="font-semibold text-neutral-900 dark:text-neutral-100">{count}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default DecisionTreeChart;