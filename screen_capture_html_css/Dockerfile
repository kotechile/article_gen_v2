FROM node:20-alpine

# Install necessary packages for Puppeteer, Chromium, and Sharp
RUN apk add --no-cache fontconfig chromium nss freetype harfbuzz ca-certificates ttf-freefont libstdc++ libx11 libxcomposite libxdamage libxext libxfixes libxi libxtst libc6-compat vips-dev

# Create fonts directory
RUN mkdir -p /usr/share/fonts/truetype

# Copy fonts from the local fonts directory
COPY fonts/* /usr/share/fonts/truetype/

# Update font cache
RUN fc-cache -f -v

# Add a non-root user
RUN addgroup -S pptruser && adduser -S pptruser -G pptruser

# Set the working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Change ownership of the application directory to pptruser
RUN chown -R pptruser:pptruser /usr/src/app

# Set environment variables
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PORT=8080

# Switch to the non-root user
USER pptruser

# Expose the port
EXPOSE 8080

# Start the application
CMD ["node", "server.js"]