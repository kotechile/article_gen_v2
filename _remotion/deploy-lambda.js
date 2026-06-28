const fs = require("fs");
const path = require("path");

// Load .env file programmatically since Remotion's Node.js API doesn't automatically read it
try {
  const envPath = path.resolve(__dirname, ".env");
  if (fs.existsSync(envPath)) {
    const envFile = fs.readFileSync(envPath, "utf8");
    envFile.split(/\r?\n/).forEach((line) => {
      // Ignore comments and empty lines
      if (line.trim().startsWith("#") || !line.trim()) return;
      const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)?\s*$/);
      if (match) {
        const key = match[1];
        let value = (match[2] || "").trim();
        // Strip wrapping quotes
        if (
          (value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))
        ) {
          value = value.slice(1, -1);
        }
        process.env[key] = value;
      }
    });
  }
} catch (e) {
  console.error("Could not load .env file:", e.message);
}

const {
  deployFunction,
  deploySite,
  getOrCreateBucket,
} = require("@remotion/lambda");
const { enableTailwind } = require("@remotion/tailwind-v4");

// Default configuration
const REGION = process.env.AWS_REGION || "us-east-1";
const MEMORY_SIZE_MB = 3008; // Adjusted to account limit (max 3008MB)
const TIMEOUT_SECONDS = 300; // 5 minutes timeout
const SITE_NAME = "artivids-engine";

async function main() {
  console.log("=== ArtiVids AWS Lambda Deployment ===");
  
  // Verify environment variables
  if (!process.env.AWS_ACCESS_KEY_ID || !process.env.AWS_SECRET_ACCESS_KEY) {
    console.warn("WARNING: AWS credentials are not set in environment variables.");
    console.warn("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to deploy successfully.");
    console.warn("Continuing under assumption that credentials are configured via ~/.aws/credentials...\n");
  }

  try {
    console.log(`[1/3] Deploying Remotion Lambda function (Region: ${REGION}, Memory: ${MEMORY_SIZE_MB}MB)...`);
    const { functionName } = await deployFunction({
      region: REGION,
      memorySizeInMb: MEMORY_SIZE_MB,
      timeoutInSeconds: TIMEOUT_SECONDS,
    });
    console.log(`✔ Function successfully deployed: ${functionName}`);

    console.log(`[2/3] Resolving S3 Bucket...`);
    const { bucketName } = await getOrCreateBucket({
      region: REGION,
    });
    console.log(`✔ S3 Bucket resolved: ${bucketName}`);

    console.log(`[3/3] Deploying site folder to S3 (Site Name: ${SITE_NAME})...`);
    const { serveUrl } = await deploySite({
      entryPoint: path.resolve(__dirname, "src/index.ts"),
      bucketName,
      region: REGION,
      siteName: SITE_NAME,
      webpackOverride: enableTailwind,
      options: {
        onBundleProgress: (progress) => {
          console.log(`   Bundling code: ${progress}%`);
        },
        onUploadProgress: ({ totalFiles, filesUploaded }) => {
          console.log(`   Uploading files: ${filesUploaded}/${totalFiles}`);
        },
      },
    });

    console.log("\n==========================================");
    console.log("✔ Deployment Successful!");
    console.log(`  Serve URL:   ${serveUrl}`);
    console.log(`  Region:      ${REGION}`);
    console.log(`  Bucket Name: ${bucketName}`);
    console.log("==========================================");
    console.log("\nTo render a vertical video on Lambda, run:");
    console.log(`npx remotion lambda render ${serveUrl} vertical --region=${REGION} --props=src/mockPayload.json`);

  } catch (error) {
    console.error("❌ Deployment failed:", error.message);
    process.exit(1);
  }
}

main();
