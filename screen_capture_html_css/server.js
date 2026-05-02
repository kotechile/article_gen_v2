const express = require('express');
const puppeteer = require('puppeteer');
const sharp = require('sharp');

const app = express(); // Define the Express app instance here

app.use(express.json());

let browser;

const initializeBrowser = async () => {
  if (!browser) {
    browser = await puppeteer.launch({
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu','--font-render-hinting=none','--force-color-profile=srgb' ],
      headless: true,
      defaultViewport: null,
    });
  }
};

app.post('/generate-image', async (req, res) => {
  const { html, css, width = 1920, height = 1080, clip, rootSelector, rootSelectors } = req.body;

  console.log('Received request:', { width, height, clip, rootSelector, rootSelectors });

  const baseCss = `
    html, body {
      margin: 0;
      padding: 0;
      background: transparent;
    }
  `;

  const fullHtml = `
    <html>
      <head>
        <link href="https://pro.fontawesome.com/releases/v6.0.0-beta1/css/all.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">        
      </head>
      <style>
        ${baseCss}
        ${css}
      </style>
      <body>
        ${html}
      </body>
    </html>
  `;

  try {
    await initializeBrowser();
    const page = await browser.newPage();

    await page.setViewport({
      width: parseInt(width),
      height: parseInt(height),
      deviceScaleFactor: 1,
    });

    // Set content and wait for network to be idle
    await page.setContent(fullHtml, { waitUntil: 'networkidle0', timeout: 60000 });

    // Wait for both <img> tags and CSS background images so the final
    // element size matches the template artwork.
    await page.evaluate(async () => {
      const imagePromises = Array.from(document.querySelectorAll('img')).map((img) => (
        new Promise((resolve) => {
          if (img.complete) {
            resolve();
            return;
          }

          img.onload = img.onerror = resolve;
        })
      ));

      const backgroundUrls = new Set();
      document.querySelectorAll('*').forEach((node) => {
        const backgroundImage = window.getComputedStyle(node).backgroundImage;
        const matches = backgroundImage.match(/url\((["']?)(.*?)\1\)/g) || [];
        matches.forEach((match) => {
          const urlMatch = match.match(/url\((["']?)(.*?)\1\)/);
          const url = urlMatch && urlMatch[2];
          if (url) backgroundUrls.add(url);
        });
      });

      const backgroundPromises = Array.from(backgroundUrls).map((url) => (
        new Promise((resolve) => {
          const img = new Image();
          img.onload = img.onerror = resolve;
          img.src = url;
        })
      ));

      await Promise.all([...imagePromises, ...backgroundPromises]);
    });

    // Additional wait to ensure all resources are loaded
    await page.waitForNetworkIdle({ timeout: 60000 });

    let imageBuffer;

    if (clip) {
      imageBuffer = await page.screenshot({
        type: 'png',
        clip: {
          x: Number(clip.x),
          y: Number(clip.y),
          width: Number(clip.width),
          height: Number(clip.height),
        },
      });
    } else if (rootSelector || (Array.isArray(rootSelectors) && rootSelectors.length > 0)) {
      const selectors = Array.isArray(rootSelectors) && rootSelectors.length > 0
        ? rootSelectors
        : [rootSelector];

      let element = null;
      let matchedSelector = null;

      for (const selector of selectors) {
        if (!selector) continue;
        element = await page.$(selector);
        if (element) {
          matchedSelector = selector;
          break;
        }
      }

      if (!element) {
        console.warn('No root selector matched; falling back to viewport screenshot.', { selectors });
        imageBuffer = await page.screenshot({ type: 'png' });
        await page.close();
        res.writeHead(200, { 'Content-Type': 'image/png' });
        res.end(imageBuffer, 'binary');
        return;
      }

      const bounds = await element.boundingBox();
      if (!bounds) {
        throw new Error(`Unable to measure root selector: ${matchedSelector}`);
      }

      const viewportWidth = Math.max(parseInt(width), Math.ceil(bounds.x + bounds.width));
      const viewportHeight = Math.max(parseInt(height), Math.ceil(bounds.y + bounds.height));

      await page.setViewport({
        width: viewportWidth,
        height: viewportHeight,
        deviceScaleFactor: 1,
      });

      imageBuffer = await element.screenshot({ type: 'png' });
    } else {
      imageBuffer = await page.screenshot({ type: 'png' });
    }

    await page.close();

    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(imageBuffer, 'binary');
  } catch (error) {
    console.error('Error generating image:', error);
    res.status(500).json({ error: 'Failed to generate image', details: error.message });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on port ${PORT}`);
});
