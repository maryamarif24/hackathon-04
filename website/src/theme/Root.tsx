/**
 * Root - Global wrapper component for Docusaurus
 */
import React from 'react';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  return <>{children}</>;
}
