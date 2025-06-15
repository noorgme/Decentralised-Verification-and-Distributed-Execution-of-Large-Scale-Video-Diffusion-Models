import { useRouter } from 'next/router';
import useSWR from 'swr';
import { Layout } from '../../components/Layout';
import { Card, CardContent, CardHeader } from 'shadcn-ui';
import { motion } from 'framer-motion';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function StatusPage() {
  const { query } = useRouter();
  const { data } = useSWR(query.requestId ? `/status/${query.requestId}` : null, fetcher, { refreshInterval: 3000 });

  return (
    <Layout>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <Card className="max-w-md mx-auto">
          <CardHeader>
            <h3 className="text-lg font-bold">Job Status: {query.requestId}</h3>
          </CardHeader>
          <CardContent>
            {data ? (
              <p>Status: <strong>{data.status}</strong></p>
            ) : (
              <p>Loading...</p>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </Layout>
  );
} 