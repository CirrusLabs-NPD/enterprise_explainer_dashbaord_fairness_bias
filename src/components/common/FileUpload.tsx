import React, { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, File, X, Check, AlertCircle, FileType, 
  Database, Image, FileText, Package
} from 'lucide-react';
import { toast } from 'react-hot-toast';

interface FileUploadProps {
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // in MB
  allowedTypes?: string[];
  onFileSelect?: (files: File[]) => void;
  onUploadComplete?: (results: any[]) => void;
  className?: string;
  disabled?: boolean;
  category?: 'model' | 'data' | 'image' | 'document';
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  preview?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  accept,
  multiple = false,
  maxSize = 100,
  allowedTypes = [],
  onFileSelect,
  onUploadComplete,
  className = '',
  disabled = false,
  category = 'data'
}) => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    switch (extension) {
      case 'csv':
      case 'json':
      case 'parquet':
      case 'xlsx':
      case 'xls':
        return <Database className="w-8 h-8 text-blue-500" />;
      case 'pkl':
      case 'joblib':
      case 'h5':
      case 'onnx':
        return <Package className="w-8 h-8 text-purple-500" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
        return <Image className="w-8 h-8 text-green-500" />;
      case 'pdf':
      case 'txt':
      case 'md':
        return <FileText className="w-8 h-8 text-orange-500" />;
      default:
        return <File className="w-8 h-8 text-neutral-500" />;
    }
  };

  const validateFile = (file: File): string | null => {
    // Check file size
    const maxSizeBytes = maxSize * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      return `File size exceeds ${maxSize}MB limit`;
    }

    // Check file type
    if (allowedTypes.length > 0) {
      const extension = file.name.split('.').pop()?.toLowerCase();
      if (!extension || !allowedTypes.includes(extension)) {
        return `File type not allowed. Allowed types: ${allowedTypes.join(', ')}`;
      }
    }

    // Category-specific validation
    const categoryTypes: Record<string, string[]> = {
      model: ['pkl', 'joblib', 'onnx', 'h5', 'pb'],
      data: ['csv', 'json', 'parquet', 'xlsx', 'xls'],
      image: ['jpg', 'jpeg', 'png', 'gif', 'bmp'],
      document: ['pdf', 'txt', 'md']
    };

    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension && !categoryTypes[category]?.includes(extension)) {
      return `Invalid file type for ${category}. Expected: ${categoryTypes[category]?.join(', ')}`;
    }

    return null;
  };

  const generatePreview = async (file: File): Promise<string | undefined> => {
    if (file.type.startsWith('image/')) {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file);
      });
    }
    return undefined;
  };

  const processFiles = async (fileList: FileList) => {
    const newFiles: UploadedFile[] = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const validation = validateFile(file);
      
      if (validation) {
        toast.error(`${file.name}: ${validation}`);
        continue;
      }

      const preview = await generatePreview(file);
      
      newFiles.push({
        file,
        id: `${Date.now()}-${i}`,
        status: 'pending',
        progress: 0,
        preview
      });
    }

    if (newFiles.length > 0) {
      setFiles(prev => multiple ? [...prev, ...newFiles] : newFiles);
      onFileSelect?.(newFiles.map(f => f.file));
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (disabled) return;
    
    processFiles(e.dataTransfer.files);
  }, [disabled, multiple]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files);
    }
  };

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const uploadFile = async (uploadedFile: UploadedFile) => {
    setFiles(prev => prev.map(f => 
      f.id === uploadedFile.id 
        ? { ...f, status: 'uploading', progress: 0 }
        : f
    ));

    try {
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise(resolve => setTimeout(resolve, 100));
        setFiles(prev => prev.map(f => 
          f.id === uploadedFile.id 
            ? { ...f, progress }
            : f
        ));
      }

      // Mock API call - replace with actual upload logic
      const formData = new FormData();
      formData.append('file', uploadedFile.file);
      formData.append('category', category);

      // Simulate API response
      const mockResponse = {
        success: true,
        filename: uploadedFile.file.name,
        size: uploadedFile.file.size,
        id: uploadedFile.id
      };

      setFiles(prev => prev.map(f => 
        f.id === uploadedFile.id 
          ? { ...f, status: 'success', progress: 100 }
          : f
      ));

      return mockResponse;
    } catch (error) {
      setFiles(prev => prev.map(f => 
        f.id === uploadedFile.id 
          ? { ...f, status: 'error', error: 'Upload failed' }
          : f
      ));
      throw error;
    }
  };

  const uploadAllFiles = async () => {
    const pendingFiles = files.filter(f => f.status === 'pending');
    const results = [];

    for (const file of pendingFiles) {
      try {
        const result = await uploadFile(file);
        results.push(result);
      } catch (error) {
        console.error(`Failed to upload ${file.file.name}:`, error);
      }
    }

    onUploadComplete?.(results);
    
    if (results.length > 0) {
      toast.success(`Successfully uploaded ${results.length} file(s)`);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200
          ${isDragOver 
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' 
            : 'border-neutral-300 dark:border-neutral-600 hover:border-primary-400'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleFileInputChange}
          className="hidden"
          disabled={disabled}
        />
        
        <motion.div
          initial={{ scale: 1 }}
          animate={{ scale: isDragOver ? 1.05 : 1 }}
          transition={{ duration: 0.2 }}
        >
          <Upload className="w-12 h-12 text-neutral-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-neutral-900 dark:text-white mb-2">
            {isDragOver ? 'Drop files here' : 'Upload files'}
          </h3>
          <p className="text-neutral-600 dark:text-neutral-400 mb-4">
            Drag and drop your files here, or click to browse
          </p>
          <div className="text-sm text-neutral-500 dark:text-neutral-400">
            <p>Maximum file size: {maxSize}MB</p>
            {allowedTypes.length > 0 && (
              <p>Allowed types: {allowedTypes.join(', ')}</p>
            )}
          </div>
        </motion.div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-neutral-900 dark:text-white">
              Selected Files ({files.length})
            </h4>
            {files.some(f => f.status === 'pending') && (
              <button
                onClick={uploadAllFiles}
                disabled={disabled}
                className="px-3 py-1 text-sm bg-primary-500 text-white rounded-md hover:bg-primary-600 disabled:opacity-50"
              >
                Upload All
              </button>
            )}
          </div>
          
          <div className="space-y-2">
            {files.map((uploadedFile) => (
              <motion.div
                key={uploadedFile.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center space-x-3 p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
              >
                {/* File Icon */}
                <div className="flex-shrink-0">
                  {uploadedFile.preview ? (
                    <img
                      src={uploadedFile.preview}
                      alt={uploadedFile.file.name}
                      className="w-10 h-10 rounded object-cover"
                    />
                  ) : (
                    getFileIcon(uploadedFile.file.name)
                  )}
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-neutral-900 dark:text-white truncate">
                    {uploadedFile.file.name}
                  </p>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    {formatFileSize(uploadedFile.file.size)}
                  </p>
                  
                  {/* Progress Bar */}
                  {uploadedFile.status === 'uploading' && (
                    <div className="mt-2">
                      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5">
                        <div
                          className="bg-primary-500 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${uploadedFile.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {uploadedFile.status === 'error' && uploadedFile.error && (
                    <p className="text-xs text-red-500 mt-1">
                      {uploadedFile.error}
                    </p>
                  )}
                </div>

                {/* Status Icon */}
                <div className="flex-shrink-0">
                  {uploadedFile.status === 'pending' && (
                    <button
                      onClick={() => uploadFile(uploadedFile)}
                      className="p-1 text-neutral-400 hover:text-primary-500"
                      title="Upload file"
                    >
                      <Upload className="w-4 h-4" />
                    </button>
                  )}
                  {uploadedFile.status === 'uploading' && (
                    <div className="animate-spin">
                      <Upload className="w-4 h-4 text-primary-500" />
                    </div>
                  )}
                  {uploadedFile.status === 'success' && (
                    <Check className="w-4 h-4 text-green-500" />
                  )}
                  {uploadedFile.status === 'error' && (
                    <AlertCircle className="w-4 h-4 text-red-500" />
                  )}
                </div>

                {/* Remove Button */}
                <button
                  onClick={() => removeFile(uploadedFile.id)}
                  className="flex-shrink-0 p-1 text-neutral-400 hover:text-red-500"
                  title="Remove file"
                >
                  <X className="w-4 h-4" />
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;