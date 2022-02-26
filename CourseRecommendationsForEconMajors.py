#!/usr/bin/env python
# coding: utf-8

# # Course Recommendations for Economics Majors
# 
# Using Yale's portal API (CoursesWebServicev2), this system looks for related courses based on course descriptions. 
# 
# Motivation
# - CourseTable and courses.yale.edu both lack recommended courses based on search. I realized that having such a recommendation system, however, would be useful. For instance, my algorithm-based recommender system course was in the economics department. Although I was able to find this class becuase I was an economics double major (who looked constantly at the economics course offerings), other CS or S&DS majors who might have been interested in these subjects might not have been able to find this class. Likewise, I encountered a Cybersecurity, Cyberwar & International Relations course that I was interested in, which was in the global affairs major. Looking through courses in all majors is nearly impossible, and for related searches, this system will hopefully make it better for students to find related courses.

# Note: This file was originally a Jupyter notebook. Urls included APIs, which were removed before being posted to GitHub.

# In[22]:


# Alternatively, this can be done using Python's requests module 
from urllib.request import urlopen
import json

# URL with API removed
url = ''

response = urlopen(url)

data_json = json.loads(response.read())

print(data_json)


# In[24]:


# Make this into pandas dataframe
import pandas as pd

econdf = pd.json_normalize(data_json)


# In[25]:


econdf.head()


# Note that the Yale API is designed so that course code is one of the required parameters and that courses for only one semester are brought if the semester is not specified. To make a more complex dataset, the next section will draw courses from other semesters. For more information on this API: https://developers.yale.edu/courseswebservicev2.

# # Description based recommendation system - for the same semester, for the same department
# 
# The recommendation system in this project will use the course description to recommend similar courses.

# In[26]:


# Import TfidfVectorizer from Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Exclude English stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Remove NaN with empty string
econdf['description'] = econdf['description'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming data
econtfidf_matrix = tfidf.fit_transform(econdf['description'])

# Output the shape of tfidf_matrix, which gives dimensions
econtfidf_matrix.shape


# In[31]:


from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
econcosine_sim = linear_kernel(econtfidf_matrix, econtfidf_matrix)


# In[32]:


econcosine_sim.shape


# This makes sense because there are 48 courses that are being offered in Spring 2022 in the Yale economics department.

# In[33]:


# Show column index 1 as example of similarity scores
# Note that row 1, column 1 is just similarity score with itself = 1
# Diagonal scores in the matrix are all 1
econcosine_sim[1]


# In[35]:


# Construct reverse map of indices and movie titles
econindices = pd.Series(econdf.index, index=econdf['courseTitle']).drop_duplicates()


# In[36]:


# Display indices for course index 0 to 9
econindices[:10]


# In[136]:


# Define the recommendation function
def get_recommendations(course, df=econdf, indices=econindices, cosine_sim=econcosine_sim):
    # Get index of course that matches the title
    idx = indices[course]
    
    # Get pairwise similarity score of all courses with that course
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort courses based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of the 5 most similar courses
    sim_scores = sim_scores[1:6]
    
    # Get course indices
    course_indices = [i[0] for i in sim_scores]
    
    # Create rows for recommendations
    rows = []
    for item in course_indices:
        rows.append([df['department'].iloc[item], df['courseNumber'].iloc[item], df['courseTitle'].iloc[item]])
    
    # Create dataframe
    column_names = ['department', 'courseNumber', 'courseTitle']
    recommends = pd.DataFrame(rows, columns=column_names)
        
    return recommends


# In[137]:


# Example - call function for the course 'Algorithms'
get_recommendations('Algorithms')


# When we searched for ECON 365 'Algorithms,' we received ECON 366 'Intensive Algorithms' as the next course that we might be interested in (ECON 365 and 366 are known as alternatives for the Algorithms course). The next course that is recommended is ECON 435 'The Role of Algorithms in the Economy,' which is also teaching algorithms in the economics department this semester. Other noticeable recommendations might be Econometrics, since this is also a quantitative reasoning-based economics course, as Algorithms is.

# # Description based recommendation system - for the same semester, across different departments
# 
# Students who are interested in taking more quantitative economics courses may be interested in taking courses from mathematics(MATH), computer science(CPSC), and statistics and data science (S&DS). On the other hand, other students may be interested in taking more qualitative economics, such as from the politics (PLSC), history (HIST), global affairs (GLBL), and sociology (SOCY). The second recommender system will combine courses that are offered in the same semester, but in different departments.

# In[191]:


# Alternatively, this can be done using Python's requests module 
# math
# URL with API removed
url1 = ''
mathdata = json.loads(response.read())


# In[193]:


# cpsc
# URL with API removed
url2 = ''
response = urlopen(url2)
cpscdata = json.loads(response.read())


# In[199]:


# s&ds
# URL with API removed
url3 = ''
sndsdata = json.loads(response.read())


# In[196]:


# plsc
# URL with API removed
url4 = ''
response = urlopen(url4)
plscdata = json.loads(response.read())


# In[200]:


# hist
# URL with API removed
url5 = ''
response = urlopen(url5)
histdata = json.loads(response.read())


# In[202]:


# glbl
# URL with API removed
url6 = ''
response = urlopen(url6)
glbldata = json.loads(response.read())


# In[203]:


# Make each into dataframe and then append them together
mathdf = pd.json_normalize(mathdata)
cpscdf = pd.json_normalize(cpscdata)
sndsdf = pd.json_normalize(sndsdata)
plscdf = pd.json_normalize(plscdata)
histdf = pd.json_normalize(histdata)
glbldf = pd.json_normalize(glbldata)


# In[204]:


# Join the dataframes
frame = [econdf, mathdf, cpscdf, sndsdf, plscdf, histdf, glbldf]
onedf = pd.DataFrame()

for df in frame: 
    onedf = onedf.append(df)


# In[205]:


# Check length of dataframe
len(onedf)


# In[206]:


# There are duplicate courses, such as ECON 110.
onedf[:10]


# In[207]:


# Remove duplicate courses from the dataframe by courseTitle
onedf = onedf.drop_duplicates(subset=['courseTitle'])
onedf[:10]


# In[208]:


# Import TfidfVectorizer from Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Exclude English stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Remove NaN with empty string
onedf['description'] = onedf['description'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming data
onetfidf_matrix = tfidf.fit_transform(onedf['description'])

# Output the shape of tfidf_matrix, which gives dimensions
onetfidf_matrix.shape


# In[209]:


from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
onecosine_sim = linear_kernel(onetfidf_matrix, onetfidf_matrix)


# In[210]:


onecosine_sim.shape


# In[211]:


oneindices = pd.Series(onedf.index, index=onedf['courseTitle']).drop_duplicates()


# In[212]:


oneindices[:10]


# Note that this displays 10 economics courses on the top because the econdf was appended to the total dataframe first.
# 
# Now we can use the recommendations function to get recommendations from this joined dataframe.

# In[190]:


# Suppose we are searching for the 'Intermediate Microeconomics' course.
get_recommendations('Intermediate Microeconomics', df=onedf, indices=oneindices, cosine_sim=onecosine_sim)


# If we search for 'Intermediate Microeconomics,' ECON 126: Macroeconomic Theory, ECON 351: Mathematical Economics: Game Theory, ECON 117: Introduction to Data Analysis and Econometrics, PLSC 332: Philosophy of Science for the Study of Politics, and ECON 375: Monetary Policy are recommended.

# In[217]:


# Suppose we are searching for the 'Algorithms' course.
get_recommendations('Algorithms', df=onedf, indices=oneindices, cosine_sim=onecosine_sim)


# If we search for a more computational course such as 'Algorithms,' the recommendation system outputs more computational courses such as ECON 366: Intensive Algorithms, CPSC 463: Algorithms via Continuous Optimization, S&DS 431: Optimization and Computation, CPSC 100: Introduction to Computing and Programming, and MATH 160: The Structure of Networks.

# In[216]:


# Suppose we search for the 'Econometrics' course.
get_recommendations('Econometrics', df=onedf, indices=oneindices, cosine_sim=onecosine_sim)


# If a student liked econometrics (which is also a quantitative reasoning(QR) economics course), they would also like ECON 409: Firms, Markets, and Competition, ECON 121: Intermediate Microeconomics, PLSC 274: Cities: Making Public Choices in New Haven,ECON 490: Immigration and its Discontents, and CPSC 472: Intelligent Robotics. The PLSC course may have been recommended because econometrics can be applied in social science research.

# # Recommendation System for Economics Majors with Past Data
# 
# This recommendation system is for economics majors who are looking for courses that they might be interested in (if they like a particular course) among courses that were historically offered in the economics department.

# In[149]:


# Bring more data from other semesters - 01: Spring, 02: Summer, 03: Fall
from urllib.request import urlopen
import json

# Spring 2022
# URL with API removed
url0 = ''
response = urlopen(url0)
sp2022 = json.loads(response.read())


# In[150]:


# Fall 2021
# URL with API removed
url1 = ''
response = urlopen(url1)
fa2021 = json.loads(response.read())


# In[145]:


# Spring 2021
# URL with API removed
url2 = ''
response = urlopen(url2)
sp2021 = json.loads(response.read())


# In[146]:


# Fall 2020
# URL with API removed
url3 = ''
response = urlopen(url3)
fa2020 = json.loads(response.read())


# In[147]:


# Spring 2020
# URL with API removed
url4 =  ''
response = urlopen(url4)
sp2020 = json.loads(response.read())


# In[154]:


# Fall 2019
# URL with API removed
url5 = ''
response = urlopen(url5)
fa2019 = json.loads(response.read())


# In[153]:


# Spring 2019
# URL with API removed
url6 = ''
response = urlopen(url6)
sp2019 = json.loads(response.read())


# In[148]:


# Fall 2018
# URL with API removed
url7 = ''
response = urlopen(url7)
fa2018 = json.loads(response.read())


# In[155]:


# Make each into dataframe and then append them together
sp22df = pd.json_normalize(sp2022)
fa21df = pd.json_normalize(fa2021)
sp21df = pd.json_normalize(sp2021)
fa20df = pd.json_normalize(fa2020)
sp20df = pd.json_normalize(sp2020)
fa19df = pd.json_normalize(fa2019)
sp19df = pd.json_normalize(sp2019)
fa18df = pd.json_normalize(fa2018)


# In[156]:


# Join the dataframes
frame = [sp22df, fa21df, sp21df, fa20df, sp20df, fa19df, sp19df, fa18df]
oneecondf = pd.DataFrame()

for df in frame: 
    oneecondf = oneecondf.append(df)


# In[157]:


# Check length of total dataframe
len(oneecondf)


# In[163]:


oneecondf[:10]


# Some economics courses are offered recurrently at Yale. Such an example is ECON 110: An Introduction to Microeconomic Analysis. There are thus duplicates when we check the first 10 rows of the dataframe above. We can choose to keep only one instance of each course by drop.duplicates() by courseTitle.

# In[165]:


# Remove duplicate courses from the dataframe by courseTitle
oneecondf = oneecondf.drop_duplicates(subset=['courseTitle'])
oneecondf[:10]


# In[171]:


# Use TfidfVectorizer from Scikit-Learn
# Exclude English stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Remove NaN with empty string
oneecondf['description'] = oneecondf['description'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming data
oneecontfidf_matrix = tfidf.fit_transform(oneecondf['description'])

# Output the shape of tfidf_matrix, which gives dimensions
oneecontfidf_matrix.shape


# In[172]:


# Compute the cosine similarity matrix
oneeconcosine_sim = linear_kernel(oneecontfidf_matrix, oneecontfidf_matrix)


# In[173]:


oneeconcosine_sim.shape


# In[178]:


oneeconindices = pd.Series(oneecondf.index, index=oneecondf['courseTitle']).drop_duplicates()
oneeconindices[:10]


# In[179]:


# Use the get_recommendations function, but with different parameters
# Suppose we are searching for the 'Intermediate Macroeconomics' course.
get_recommendations('Intermediate Macroeconomics', df=oneecondf, indices=oneeconindices, cosine_sim=oneeconcosine_sim)


# If a student liked 'Intermediate Macroeconomics,' data from the past 8 semesters suggests that the student will also like ECON 477: Topics in the Economics of Education, ECON 002: Social Issues in America, ECON 210: Economics of Education, ECON 160: Applications of Game Theory, and ECON 170: Health Economics and Public Policy.
# 
# Note that this data is collected from 8 previous semesters at Yale, and that not all economics courses are offered each year. If this recommendation system is to be used, students should check whether or not each course is being offered. Nevertheless, this system can give the student a broader idea of the topics they might potentially be interested in.

# In[180]:


# Suppose we are searching for the 'Econometrics' course.
get_recommendations('Econometrics', df=oneecondf, indices=oneeconindices, cosine_sim=oneeconcosine_sim)


# If a student liked 'ECON 136: Econometrics,' the student might also like ECON 409: Firms, Markets, and Competition, ECON 121: Intermediate Microeconomics, ECON 431: Economics and Psychology, ECON 275: Public Economics, and ECON 414: Economic Models of New Technology.

# In[181]:


# Suppose we are searching for the 'Algorithms' course.
get_recommendations('Algorithms', df=oneecondf, indices=oneeconindices, cosine_sim=oneeconcosine_sim)


# If a student was looking for a more computational and quantitative course, such as Algorithms, the recommendation system gives more computational economic courses, such as ECON 366: Intensive Algorithms, ECON 425: Economics and Computation, ECON 413: Optimization Techniques, ECON 435: The Role of Algorithms in the Economy, and ECON 135: Introduction to Probability and Statistics.
