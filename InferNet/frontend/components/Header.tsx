import React from 'react';
import Link from 'next/link';
import { ConnectButton } from '@rainbow-me/rainbowkit';

export function Header() {
  return (
    <header className="bg-white shadow">
      <div className="container mx-auto flex items-center justify-between p-4">
        <h1 className="text-2xl font-bold text-gray-800">
          <Link href="/">InferNet</Link>
        </h1>
        <nav className="space-x-4">
          <Link href="/">Home</Link>
          <Link href="/status">Status</Link>
        </nav>
        <ConnectButton />
      </div>
    </header>
  );
} 