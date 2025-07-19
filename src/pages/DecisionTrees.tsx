import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { GitBranch, Layers, TrendingUp, Download, Settings, Filter, BarChart3, Eye } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import DecisionTreeChart from '@/components/charts/DecisionTreeChart';

interface DecisionTreesProps {
  modelType: 'classification' | 'regression';
}

interface TreeNode {
  id: string;
  feature?: string;
  threshold?: number;
  value?: number;
  samples: number;
  impurity: number;
  left?: TreeNode;
  right?: TreeNode;
  isLeaf: boolean;
  depth: number;
  prediction?: number;
  class?: string;
}

interface TreeStats {
  totalNodes: number;
  leafNodes: number;
  maxDepth: number;
  avgDepth: number;
  totalSamples: number;
}

const DecisionTrees: React.FC<DecisionTreesProps> = ({ modelType }) => {
  const [selectedTreeIndex, setSelectedTreeIndex] = useState(0);
  const [maxDepth, setMaxDepth] = useState(3);
  const [highlightedPath, setHighlightedPath] = useState<string[]>([]);
  // const [selectedInstance, setSelectedInstance] = useState<Record<string, number>>({});
  const [viewMode, setViewMode] = useState<'tree' | 'rules' | 'stats'>('tree');

  // Mock tree data generator
  const generateTree = (depth: number, maxDepth: number, nodeId: string = '0'): TreeNode => {
    const isLeaf = depth >= maxDepth || Math.random() < 0.3;
    const samples = Math.floor(Math.random() * 1000) + 100;
    const impurity = Math.random() * 0.5;
    
    if (isLeaf) {
      return {
        id: nodeId,
        samples,
        impurity,
        isLeaf: true,
        depth,
        prediction: modelType === 'classification' ? Math.random() : Math.random() * 100,
        class: modelType === 'classification' ? (Math.random() > 0.5 ? 'Approved' : 'Rejected') : undefined,
      };
    }
    
    const features = ['Annual_Income', 'Credit_Score', 'Customer_Age', 'Account_Balance', 'Loan_Amount'];
    const feature = features[Math.floor(Math.random() * features.length)];
    const threshold = Math.random() * 100 + 50;
    
    return {
      id: nodeId,
      feature,
      threshold,
      samples,
      impurity,
      isLeaf: false,
      depth,
      left: generateTree(depth + 1, maxDepth, `${nodeId}L`),
      right: generateTree(depth + 1, maxDepth, `${nodeId}R`),
    };
  };

  const trees = useMemo(() => {
    return Array.from({ length: 5 }, (_, i) => ({
      id: i,
      name: `Tree ${i + 1}`,
      accuracy: 0.85 + Math.random() * 0.1,
      importance: 0.15 + Math.random() * 0.1,
      tree: generateTree(0, maxDepth),
    }));
  }, [maxDepth]);

  const selectedTree = trees[selectedTreeIndex];

  // Helper function to convert TreeNode to DecisionNode format
  const convertTreeNode = (node: TreeNode, depth: number): any => {
    if (!node) return undefined;
    
    return {
      id: node.id,
      type: node.isLeaf ? 'leaf' : 'decision',
      feature: node.feature,
      threshold: node.threshold,
      operator: '<=',
      samples: node.samples,
      purity: 1 - node.impurity,
      prediction: node.prediction || node.class,
      confidence: 0.85,
      children: node.isLeaf ? undefined : [
        node.left && convertTreeNode(node.left, depth + 1),
        node.right && convertTreeNode(node.right, depth + 1)
      ].filter(Boolean),
      depth,
      gini: node.impurity,
      entropy: node.impurity * 0.8,
      classDistribution: modelType === 'classification' ? {
        'Approved': Math.floor(node.samples * 0.6),
        'Rejected': Math.floor(node.samples * 0.4)
      } : undefined
    };
  };

  const calculateTreeStats = (node: TreeNode): TreeStats => {
    const traverse = (n: TreeNode): { nodes: number; leaves: number; depths: number[] } => {
      if (n.isLeaf) {
        return { nodes: 1, leaves: 1, depths: [n.depth] };
      }
      
      const leftStats = n.left ? traverse(n.left) : { nodes: 0, leaves: 0, depths: [] };
      const rightStats = n.right ? traverse(n.right) : { nodes: 0, leaves: 0, depths: [] };
      
      return {
        nodes: 1 + leftStats.nodes + rightStats.nodes,
        leaves: leftStats.leaves + rightStats.leaves,
        depths: [...leftStats.depths, ...rightStats.depths],
      };
    };
    
    const stats = traverse(node);
    return {
      totalNodes: stats.nodes,
      leafNodes: stats.leaves,
      maxDepth: Math.max(...stats.depths),
      avgDepth: stats.depths.reduce((a, b) => a + b, 0) / stats.depths.length,
      totalSamples: node.samples,
    };
  };

  const treeStats = calculateTreeStats(selectedTree.tree);

  const renderTreeNode = (node: TreeNode, x: number, y: number, parentX?: number, parentY?: number) => {
    const nodeSize = 40;
    const isHighlighted = highlightedPath.includes(node.id);
    
    return (
      <g key={node.id}>
        {/* Edge to parent */}
        {parentX !== undefined && parentY !== undefined && (
          <motion.line
            initial={{ opacity: 0, pathLength: 0 }}
            animate={{ opacity: 1, pathLength: 1 }}
            transition={{ delay: node.depth * 0.2 }}
            x1={parentX}
            y1={parentY}
            x2={x}
            y2={y}
            stroke={isHighlighted ? '#3b82f6' : '#6b7280'}
            strokeWidth={isHighlighted ? 3 : 2}
            className="transition-all duration-200"
          />
        )}
        
        {/* Node */}
        <motion.g
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: node.depth * 0.2 + 0.1 }}
          className="cursor-pointer"
          onClick={() => {
            const path = [];
            const current = node;
            while (current) {
              path.push(current.id);
              // In a real implementation, you'd traverse up to parent
              break;
            }
            setHighlightedPath(path);
          }}
        >
          <circle
            cx={x}
            cy={y}
            r={nodeSize / 2}
            fill={node.isLeaf ? '#22c55e' : '#3b82f6'}
            stroke={isHighlighted ? '#1d4ed8' : '#374151'}
            strokeWidth={isHighlighted ? 3 : 2}
            className="transition-all duration-200 hover:stroke-primary-500"
          />
          
          {/* Node label */}
          <text
            x={x}
            y={y - 5}
            textAnchor="middle"
            className="text-xs font-medium fill-white pointer-events-none"
          >
            {node.isLeaf ? (node.class || node.prediction?.toFixed(1)) : node.feature?.substring(0, 8)}
          </text>
          
          {!node.isLeaf && (
            <text
              x={x}
              y={y + 8}
              textAnchor="middle"
              className="text-xs fill-white pointer-events-none"
            >
              ≤ {node.threshold?.toFixed(1)}
            </text>
          )}
        </motion.g>
        
        {/* Recursively render children */}
        {!node.isLeaf && (
          <>
            {node.left && renderTreeNode(node.left, x - 80, y + 80, x, y)}
            {node.right && renderTreeNode(node.right, x + 80, y + 80, x, y)}
          </>
        )}
      </g>
    );
  };

  const generateDecisionRules = (node: TreeNode, path: string[] = []): string[] => {
    if (node.isLeaf) {
      const condition = path.join(' AND ');
      const prediction = node.class || node.prediction?.toFixed(2);
      return [`IF ${condition} THEN ${prediction}`];
    }
    
    const rules: string[] = [];
    if (node.left) {
      rules.push(...generateDecisionRules(node.left, [...path, `${node.feature} <= ${node.threshold?.toFixed(2)}`]));
    }
    if (node.right) {
      rules.push(...generateDecisionRules(node.right, [...path, `${node.feature} > ${node.threshold?.toFixed(2)}`]));
    }
    
    return rules;
  };

  const decisionRules = generateDecisionRules(selectedTree.tree);

  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="initial"
      animate="animate"
      className="space-y-6"
    >
      <motion.div variants={itemVariants} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Decision Trees
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Visualize and analyze decision tree structures for tree-based models
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
          >
            Settings
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Export
          </Button>
        </div>
      </motion.div>

      {/* Tree Selection and Controls */}
      <motion.div variants={itemVariants}>
        <Card
          title="Tree Selection & Controls"
          icon={<Filter className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {trees.map((tree, index) => (
                <Button
                  key={tree.id}
                  variant={selectedTreeIndex === index ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedTreeIndex(index)}
                  className="flex flex-col items-center h-auto p-3"
                >
                  <div className="font-medium">{tree.name}</div>
                  <div className="text-xs opacity-70">Acc: {(tree.accuracy * 100).toFixed(1)}%</div>
                </Button>
              ))}
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 items-center">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Max Depth:</span>
                <input
                  type="range"
                  min="2"
                  max="6"
                  step="1"
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                  className="w-24"
                />
                <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 min-w-8">
                  {maxDepth}
                </span>
              </div>
              
              <div className="flex items-center space-x-2">
                {[
                  { id: 'tree', label: 'Tree View', icon: <GitBranch className="w-4 h-4" /> },
                  { id: 'rules', label: 'Decision Rules', icon: <BarChart3 className="w-4 h-4" /> },
                  { id: 'stats', label: 'Statistics', icon: <TrendingUp className="w-4 h-4" /> },
                ].map((mode) => (
                  <Button
                    key={mode.id}
                    variant={viewMode === mode.id ? 'primary' : 'outline'}
                    size="sm"
                    leftIcon={mode.icon}
                    onClick={() => setViewMode(mode.id as any)}
                  >
                    {mode.label}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Enhanced Decision Tree Visualization */}
      <motion.div variants={itemVariants}>
        <DecisionTreeChart
          treeData={{
            id: selectedTree.tree.id,
            type: selectedTree.tree.isLeaf ? 'leaf' : 'decision',
            feature: selectedTree.tree.feature,
            threshold: selectedTree.tree.threshold,
            operator: '<=',
            samples: selectedTree.tree.samples,
            purity: 1 - selectedTree.tree.impurity,
            prediction: selectedTree.tree.prediction || selectedTree.tree.class,
            confidence: 0.85,
            children: selectedTree.tree.isLeaf ? undefined : [
              convertTreeNode(selectedTree.tree.left!, 1),
              convertTreeNode(selectedTree.tree.right!, 1)
            ],
            depth: 0,
            gini: selectedTree.tree.impurity,
            entropy: selectedTree.tree.impurity * 0.8,
            classDistribution: modelType === 'classification' ? {
              'Approved': Math.floor(selectedTree.tree.samples * 0.6),
              'Rejected': Math.floor(selectedTree.tree.samples * 0.4)
            } : undefined
          }}
          maxDepth={maxDepth}
          modelType={modelType}
          interactive={true}
          onNodeClick={(node) => {
            console.log('Selected node:', node);
            // Update highlighted path logic
          }}
          highlightPath={highlightedPath}
          showMetrics={true}
        />
      </motion.div>

      {/* Main Visualization */}
      <motion.div variants={itemVariants}>
        <Card
          title={`${selectedTree.name} - ${viewMode === 'tree' ? 'Tree Structure' : viewMode === 'rules' ? 'Decision Rules' : 'Tree Statistics'}`}
          icon={<Eye className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          {viewMode === 'tree' && (
            <div className="space-y-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Enhanced interactive decision tree visualization above. Traditional view below for comparison.
              </div>
              
              <div className="relative h-96 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-4 overflow-auto">
                <svg viewBox="0 0 800 400" className="w-full h-full min-w-[800px]">
                  {renderTreeNode(selectedTree.tree, 400, 50)}
                </svg>
              </div>
              
              <div className="flex items-center justify-center space-x-6 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Decision Node</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-green-500"></div>
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Leaf Node</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-0.5 bg-blue-500"></div>
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Highlighted Path</span>
                </div>
              </div>
            </div>
          )}
          
          {viewMode === 'rules' && (
            <div className="space-y-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Decision rules extracted from the tree structure ({decisionRules.length} rules total)
              </div>
              
              <div className="max-h-96 overflow-y-auto space-y-2">
                {decisionRules.map((rule, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
                  >
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-xs font-bold text-primary-600 dark:text-primary-400">
                          {index + 1}
                        </span>
                      </div>
                      <div className="flex-1 font-mono text-sm text-neutral-900 dark:text-neutral-100 leading-relaxed">
                        {rule}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
          
          {viewMode === 'stats' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {treeStats.totalNodes}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Total Nodes
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {treeStats.leafNodes}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Leaf Nodes
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {treeStats.maxDepth}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Max Depth
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {treeStats.avgDepth.toFixed(1)}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Avg Depth
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Tree Performance
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Accuracy:</span>
                      <span className="font-bold text-primary-600 dark:text-primary-400">
                        {(selectedTree.accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Importance:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {(selectedTree.importance * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Total Samples:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {treeStats.totalSamples.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Tree Complexity
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Node Ratio:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {(treeStats.leafNodes / treeStats.totalNodes * 100).toFixed(1)}% leaves
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Branching:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {((treeStats.totalNodes - treeStats.leafNodes) / treeStats.leafNodes).toFixed(2)} splits/leaf
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Depth Efficiency:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {(treeStats.avgDepth / treeStats.maxDepth * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Card>
      </motion.div>

      {/* Tree Comparison */}
      <motion.div variants={itemVariants}>
        <Card
          title="Tree Comparison"
          icon={<Layers className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="space-y-4">
            <div className="text-sm text-neutral-600 dark:text-neutral-400">
              Compare performance and complexity across different trees in the ensemble
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-neutral-200 dark:border-neutral-700">
                    <th className="text-left p-2 text-neutral-900 dark:text-neutral-100">Tree</th>
                    <th className="text-right p-2 text-neutral-900 dark:text-neutral-100">Accuracy</th>
                    <th className="text-right p-2 text-neutral-900 dark:text-neutral-100">Importance</th>
                    <th className="text-right p-2 text-neutral-900 dark:text-neutral-100">Nodes</th>
                    <th className="text-right p-2 text-neutral-900 dark:text-neutral-100">Depth</th>
                    <th className="text-right p-2 text-neutral-900 dark:text-neutral-100">Leaves</th>
                  </tr>
                </thead>
                <tbody>
                  {trees.map((tree, index) => {
                    const stats = calculateTreeStats(tree.tree);
                    return (
                      <motion.tr
                        key={tree.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors cursor-pointer ${
                          selectedTreeIndex === index ? 'bg-primary-50 dark:bg-primary-900/20' : ''
                        }`}
                        onClick={() => setSelectedTreeIndex(index)}
                      >
                        <td className="p-2 font-medium text-neutral-900 dark:text-neutral-100">
                          {tree.name}
                        </td>
                        <td className="p-2 text-right text-neutral-900 dark:text-neutral-100">
                          {(tree.accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-right text-neutral-900 dark:text-neutral-100">
                          {(tree.importance * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-right text-neutral-900 dark:text-neutral-100">
                          {stats.totalNodes}
                        </td>
                        <td className="p-2 text-right text-neutral-900 dark:text-neutral-100">
                          {stats.maxDepth}
                        </td>
                        <td className="p-2 text-right text-neutral-900 dark:text-neutral-100">
                          {stats.leafNodes}
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default DecisionTrees;