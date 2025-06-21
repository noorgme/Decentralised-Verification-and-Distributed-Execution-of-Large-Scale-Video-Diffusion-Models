import { useEffect } from 'react'

export default function DebugPage() {
  useEffect(() => {
    console.log(' Environment Variables Debug:')
    console.log('  NEXT_PUBLIC_CONTRACT_ADDRESS:', process.env.NEXT_PUBLIC_CONTRACT_ADDRESS)
    console.log('  NODE_ENV:', process.env.NODE_ENV)
    console.log('  All env vars:', Object.keys(process.env).filter(key => key.startsWith('NEXT_PUBLIC_')))
  }, [])

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>Environment Variables Debug</h1>
      <div>
        <strong>NEXT_PUBLIC_CONTRACT_ADDRESS:</strong> {process.env.NEXT_PUBLIC_CONTRACT_ADDRESS || 'NOT SET'}
      </div>
      <div>
        <strong>NODE_ENV:</strong> {process.env.NODE_ENV || 'NOT SET'}
      </div>
      <div>
        <strong>All NEXT_PUBLIC_ vars:</strong>
        <ul>
          {Object.keys(process.env)
            .filter(key => key.startsWith('NEXT_PUBLIC_'))
            .map(key => (
              <li key={key}>{key}: {process.env[key]}</li>
            ))}
        </ul>
      </div>
    </div>
  )
} 