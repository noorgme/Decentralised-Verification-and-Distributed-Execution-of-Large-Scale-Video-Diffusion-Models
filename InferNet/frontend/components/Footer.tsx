import React from 'react';

export function Footer() {
  return (
    <footer className="bg-white mt-12 py-4">
      <div className="container mx-auto text-center text-gray-600">
        © {new Date().getFullYear()} InferNet. All rights reserved.
      </div>
    </footer>
  );
} 