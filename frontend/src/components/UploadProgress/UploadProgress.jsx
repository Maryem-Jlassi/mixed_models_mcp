import React from 'react';
import { FileText } from 'lucide-react';
import './UploadProgress.css';

const UploadProgress = ({ fileName }) => {
  return (
    <div className="upload-progress-container">
      <div className="file-upload-icon">
        <FileText size={48} className="uploading-file-icon" />
      </div>
      <p className="file-name">{fileName}</p>
    </div>
  );
};

export default UploadProgress;
