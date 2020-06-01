from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from nilearn.input_data import NiftiMasker
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from nilearn import datasets
from sklearn.svm import SVC
import pandas as pd


haxby_dataset = datasets.fetch_haxby()

fmri_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask_vt[0]

masker = NiftiMasker(mask_img=mask_filename, standardize=True)

behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

conditions = behavioral['labels']

fmri_masked = masker.fit_transform(fmri_filename)

fmri_train, fmri_test, conditions_train, conditions_test = train_test_split(fmri_masked, conditions, test_size=0.2, random_state=0)


svc = SVC(kernel='linear')
svc.fit(fmri_train,conditions_train)
svm_prediction = svc.predict(fmri_test)
svm_accuracy=accuracy_score(conditions_test,svm_prediction)
print(svm_accuracy)

gnb = GaussianNB()
gnb.fit(fmri_train, conditions_train)
gnb_prediction = gnb.predict(fmri_test)
gnb_accuracy=accuracy_score(conditions_test,gnb_prediction)
print(gnb_accuracy)

kneigh = KNeighborsClassifier(n_neighbors=3)
kneigh.fit(fmri_train, conditions_train)
kneigh_prediction = kneigh.predict(fmri_test)
kneigh_accuracy=accuracy_score(conditions_test,kneigh_prediction)
print(kneigh_accuracy)

gmm=GaussianMixture(n_components=3, covariance_type='spherical', max_iter=10)
gmm.fit(fmri_train, conditions_train)
gmm_prediction = gmm.predict(fmri_test)
gmm_accuracy=accuracy_score(conditions_test,gmm_prediction)
print(gmm_accuracy)
