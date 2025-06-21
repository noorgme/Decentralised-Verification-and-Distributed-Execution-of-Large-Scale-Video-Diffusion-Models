import { ethers } from "hardhat";

async function main() {
  console.log("Deploying InferNet contracts...");

  // Deploy Mock TAO token first
  console.log("Deploying Mock TAO token...");
  const MockTAO = await ethers.getContractFactory("MockTAO");
  const mockTAO = await MockTAO.deploy();
  await mockTAO.waitForDeployment();
  const taoAddress = await mockTAO.getAddress();
  console.log(`Mock TAO deployed to: ${taoAddress}`);

  // Deploy InferNetRewards contract
  console.log("Deploying InferNetRewards contract...");
  const InferNetRewards = await ethers.getContractFactory("InferNetRewards");

  // Placeholder for demo
  const validatorAddress = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"; // Anvil account 1

  const contract = await InferNetRewards.deploy(validatorAddress, taoAddress);
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log(`InferNetRewards deployed to: ${address}`);

  // Transfer some TAO tokens to the contract for testing
  const transferAmount = ethers.parseEther("10000"); // 10k TAO
  await mockTAO.transfer(address, transferAmount);
  console.log(`Transferred ${ethers.formatEther(transferAmount)} TAO to contract`);

  // Save deployment info
  const deploymentInfo = {
    contractAddress: address,
    taoTokenAddress: taoAddress,
    validatorAddress: validatorAddress,
    network: "localhost",
    timestamp: new Date().toISOString()
  };

  console.log("Deployment info:", deploymentInfo);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
