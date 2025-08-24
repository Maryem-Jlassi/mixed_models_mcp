import React from 'react';
import ReactDOM from 'react-dom/client';
import ContextProvider from './context/Context';
import App from './App';
import './index.css';
import reportWebVitals from './reportWebVitals';

console.log('Starting app...');

const rootElement = document.getElementById('root');

if (!rootElement) {
  console.error('Failed to find the root element');
} else {
  console.log('Root element found:', rootElement);
  
  const root = ReactDOM.createRoot(rootElement);
  
  try {
    console.log('Rendering app...');
    root.render(
      <React.StrictMode>
        <ContextProvider>
          <App />
        </ContextProvider>
      </React.StrictMode>
    );
    console.log('App rendered successfully');
  } catch (error) {
    console.error('Error rendering app:', error);
  }
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
