/* pages/index.tsx */
import { useState } from 'react'
import Layout            from '../components/Layout'
import { Button }        from '@/components/ui/button'
import {
  useWriteContract,
  useAccount,
}                         from 'wagmi'
import {
  parseEther,
  keccak256,
  type Address,
  type Hex,
}                         from 'viem'
import { customAlphabet } from 'nanoid'
import CONTRACT_ABI      from '../contracts/InferNetRewards.abi.json'
import { mainnet } from 'wagmi/chains'
import { stringToBytes } from 'viem'

const CONTRACT_ADDRESS = process.env.NEXT_PUBLIC_CONTRACT_ADDRESS as Address

export default function Home () {
  const [prompt,    setPrompt]    = useState('')
  const [jobId,     setJobId]     = useState<bigint | null>(null)
  const [status,    setStatus]    = useState<string | null>(null)

  const { writeContractAsync, isPending } = useWriteContract()
  const { address: account }               = useAccount()

  //  helpers
  
  
  // *** flat protocol fee  for test
  const FIXED_DEPOSIT_ETH = '0.02'                // string keeps TS happy
  const FIXED_DEPOSIT_WEI = parseEther(FIXED_DEPOSIT_ETH)
  //  actions
  async function handleDeposit () {
    if (!prompt.trim()) {
      alert('Enter a prompt 🙂')
      return
    }
    

    try {
      // 1. generate deterministic job id
      const hexNanoid = customAlphabet('0123456789abcdef', 16) // 64-bit entropy
      const reqId     = BigInt('0x' + hexNanoid())   
      setJobId(reqId)

      // 2. on-chain commit-and-deposit
      setStatus('Submitting on-chain deposit ')
      const txHash: Hex = await writeContractAsync({
        address:       CONTRACT_ADDRESS,
        abi:           CONTRACT_ABI,
        functionName: 'depositAndCommit',
        args: [reqId, keccak256(stringToBytes(prompt)), FIXED_DEPOSIT_WEI],
        overrides: {
          gasLimit: 300_000n
        }
      })
      setStatus(`Tx sent: ${txHash.slice(0, 10)} waiting for confirm`)

      // 3. off-chain prompt delivery
      await fetch('/submit_prompt', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ request_id: reqId.toString(), prompt }),
      })
      setStatus('Prompt submitted awaiting validators')
    } catch (err) {
      console.error(err)
      setStatus('error during deposit')
    }
  }

  //  UI
  return (
    <Layout>
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-blue-100 py-12 px-4">
        <div className="w-full max-w-xl bg-white rounded-2xl shadow-2xl p-10 space-y-8 border border-blue-100">
          <h2 className="text-3xl font-extrabold text-center text-blue-700 mb-2 tracking-tight">Generate a Video</h2>
          <p className="text-center text-gray-500 mb-6">Describe your video prompt and deposit ETH to start the job.</p>

          {/* prompt  --------------------------------------------------- */}
          <label className="block space-y-2">
            <span className="text-base font-semibold text-gray-700">Prompt</span>
            <textarea
              rows={4}
              className="mt-1 w-full p-3 border-2 border-blue-200 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition resize-none text-gray-800 bg-blue-50 placeholder-gray-400"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="A red-panda astronaut surfing the rings of Saturn "
            />
          </label>

          {/* amount  --------------------------------------------------- */}
          {/* fixed price banner --------------------------------------- */}
          <p className="text-center text-sm text-blue-600">
            Deposit required: <b>{FIXED_DEPOSIT_ETH} ETH</b>
          </p>

          {/* action  --------------------------------------------------- */}
          <Button
            onClick={handleDeposit}
            disabled={isPending}
            className="w-full py-3 text-lg font-bold rounded-xl bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-500 text-white shadow-lg hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 border-none focus:outline-none focus:ring-4 focus:ring-blue-300"
          >
            {isPending ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path></svg>
                Waiting
              </span>
            ) : 'Deposit & Generate'}
          </Button>

          {/* informational  ------------------------------------------- */}
          {status && (
            <div className={`text-center text-base font-medium py-3 rounded-lg ${status.startsWith('error') ? 'bg-red-100 text-red-700' : status.includes('✓') ? 'bg-green-100 text-green-700' : 'bg-blue-50 text-blue-700'} shadow-sm mt-2`}>{status}</div>
          )}

          {jobId !== null && (
            <div className="text-xs text-center text-gray-400 break-all mt-2">
              <span className="font-semibold text-gray-500">Job ID:</span> <span className="font-mono">{jobId.toString(16)}</span>
            </div>
          )}
        </div>
      </div>
    </Layout>
  )
}
