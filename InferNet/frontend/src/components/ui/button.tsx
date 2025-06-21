import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onClick?: () => void;
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  className = '', 
  disabled = false, 
  onClick,
  ...props 
}) => {
  return (
    <button
      className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;
