import { useState } from 'react';
import { useRouter } from 'next/router';
import { Layout } from '../../components/Layout';
import { Button } from 'shadcn-ui';

export default function StatusIndex() {
  const [requestId, setRequestId] = useState('');
  const router = useRouter();

  function goToStatus() {
    if (requestId) router.push(`/status/${requestId}`);
  }

  return (
    <Layout>
      <div className="max-w-md mx-auto mt-12 space-y-6">
        <h2 className="text-xl font-semibold">Check Job Status</h2>
        <input
          className="w-full p-2 border rounded"
          placeholder="Enter Request ID"
          value={requestId}
          onChange={e => setRequestId(e.target.value)}
        />
        <Button onClick={goToStatus} className="w-full">Check Status</Button>
      </div>
    </Layout>
  );
} 