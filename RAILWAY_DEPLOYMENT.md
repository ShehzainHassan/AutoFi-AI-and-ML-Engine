# Railway Deployment Guide for AutoFi Vehicle Recommendation System

## Prerequisites
- Railway account (https://railway.app)
- GitHub repository with your project
- PostgreSQL database (Railway provides this)

## 🚀 Deployment Steps

### 1. **Connect to Railway**
1. Go to [Railway](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `AutoFi-Vehicle-Recommendation-System` repository

### 2. **Add PostgreSQL Database**
1. In your Railway project dashboard, click "New Service"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically provision a PostgreSQL database

### 3. **Configure Environment Variables**
In your Railway project settings, add these environment variables:

#### **Required Variables:**
```bash
# These are automatically provided by Railway PostgreSQL service
DATABASE_URL=postgresql://user:password@host:port/database

# Application Configuration
MODEL_PATH=trained_models/
MAX_RECOMMENDATIONS=10
ENVIRONMENT=production

# Optional (Railway auto-configures these)
PORT=8000
HOST=0.0.0.0
```

#### **Database Variables (Alternative - if DATABASE_URL doesn't work):**
```bash
DB_HOST=your-railway-postgres-host
DB_NAME=railway
DB_USER=postgres
DB_PASSWORD=your-railway-postgres-password
DB_PORT=5432
```

### 4. **Deployment Configuration**
Railway will automatically detect your Python project and use the files we created:
- `Procfile` - Tells Railway how to start your app
- `railway.toml` - Railway-specific configuration
- `requirements.txt` - Python dependencies

### 5. **Start Command Options**
Railway will use one of these (in order of preference):
1. `Procfile`: `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
2. `start.py`: `python start.py`
3. Direct: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## 🔧 Key Files Created/Modified

### **New Files:**
- `Procfile` - Railway startup command
- `railway.toml` - Railway configuration
- `start.py` - Alternative startup script
- `RAILWAY_DEPLOYMENT.md` - This guide

### **Modified Files:**
- `config.py` - Railway-friendly configuration
- `app/db.py` - Better database connection handling

## 🗄️ Database Setup

### **Important Notes:**
1. **ML Models**: Your `trained_models/` directory will be deployed with your app
2. **Database Schema**: You'll need to create your database tables after deployment
3. **Data Migration**: Consider how to populate your database with initial data

### **Database Connection:**
Railway automatically provides `DATABASE_URL` in this format:
```
postgresql://postgres:password@host:port/railway
```

## 🚀 Deployment Process

1. **Push to GitHub**: Make sure all changes are committed and pushed
2. **Railway Auto-Deploy**: Railway will automatically build and deploy
3. **Monitor Logs**: Check Railway logs for any deployment issues
4. **Test API**: Access your deployed API at the Railway-provided URL

## 🔍 Testing Your Deployment

After deployment, test these endpoints:
- `GET /` - Health check
- `GET /api/recommendations/user/{user_id}` - User recommendations
- `GET /api/recommendations/similar/{vehicle_id}` - Similar vehicles
- `GET /api/recommendations/interactions-summary` - Interactions data
- `GET /api/recommendations/vehicle-features` - Vehicle features

## 🐛 Troubleshooting

### **Common Issues:**

1. **Database Connection Failed:**
   - Check `DATABASE_URL` is set correctly
   - Verify PostgreSQL service is running
   - Check logs for specific connection errors

2. **ML Models Not Loading:**
   - Ensure `trained_models/` directory is in your repository
   - Check `MODEL_PATH` environment variable
   - Verify file permissions

3. **Port Issues:**
   - Railway automatically sets `PORT` environment variable
   - Don't hardcode port numbers

4. **Memory Issues:**
   - ML models might be large - consider Railway's memory limits
   - Monitor resource usage in Railway dashboard

### **Debugging Commands:**
```bash
# Check Railway logs
railway logs

# Connect to your database
railway connect

# Check environment variables
railway variables
```

## 🔐 Security Considerations

1. **Environment Variables**: Never commit `.env` files
2. **Database**: Use Railway's provided PostgreSQL service
3. **API Keys**: Store sensitive data in Railway environment variables
4. **CORS**: Configure CORS properly for your frontend domain

## 📈 Performance Tips

1. **Database Indexing**: Add indexes for frequently queried columns
2. **Caching**: Consider Redis for caching recommendations
3. **ML Model Optimization**: Optimize model loading for faster startup
4. **Monitoring**: Use Railway's built-in monitoring tools

## 🔄 Continuous Deployment

Railway automatically redeploys when you push to your main branch:
1. Make changes locally
2. Commit and push to GitHub
3. Railway automatically builds and deploys
4. Monitor deployment logs

## 📞 Support

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **GitHub Issues**: Create issues in your repository

---

**Happy Deploying! 🚄** 