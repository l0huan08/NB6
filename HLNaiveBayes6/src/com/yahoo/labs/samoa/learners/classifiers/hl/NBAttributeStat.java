package com.yahoo.labs.samoa.learners.classifiers.hl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yahoo.labs.samoa.core.ContentEvent;
import com.yahoo.labs.samoa.core.Processor;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.learners.InstanceContentEvent;
import com.yahoo.labs.samoa.learners.InstancesContentEvent;
import com.yahoo.labs.samoa.learners.classifiers.hl.attributeclassobservers.AttributeClassObserver;
import com.yahoo.labs.samoa.learners.classifiers.hl.attributeclassobservers.GaussianNumericAttributeClassObserver;
import com.yahoo.labs.samoa.learners.classifiers.hl.attributeclassobservers.NominalAttributeClassObserver;
import com.yahoo.labs.samoa.topology.Stream;

/**
 * The processor to statistic P(Xi|Cj) for each attribute i, class j.
 * In training phase:
 * Each NBAttributeStat processor observe the training data from trainStream,
 * update P(Xi|Cj) for all classes, and for a range of attribtues iMin<=i<=iMax
 * 
 * also record the nTrain, the total number of train instances
 * and nTrain_j, the total number of train instances that is class j
 * 
 * update the NBModels
 * 
 * In testing phase:
 * do nothing
 * 
 * @author hl
 * 2014.4.16
 * edit 2014.4.18 change topology, the testing task is moved to NBModel
 * after training, also need to update the NBModel
 */
public class NBAttributeStat implements Processor {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4096870375226345888L;
	private static Logger logger = LoggerFactory.getLogger(NBAttributeStat.class);
	
	private int processorId;
	
	// range of attribute index, iMin<=i<=iMax
	private int iMin=0;
	private int iMax=0;
	private int iRange=0;//iRange=iMax-iMin+1
	private int K=0; //number of classes of dataset
	
	private int[] nTrain_j; //totall number of train instances that is class j
	private int nTrain=0; //totall number of train instances. -2,147,483,648 to 2,147,483,647
	
	// number of attributes
	//private int M;
	
	// number of NBAttributeStat processors in whole topology
	private int p1=1;
	
	// update NBModel after training every F instances
	private int updateFrequency;
	private int nTrainedInstancesBeforeUpdate; //num of trained instance in current training cycle
	
	
	//required parameters
	private final Instances dataset; //only use in builder
	private InstancesHeader header; //attribute information

	//private Stream trainStream;
	//private Stream testStream;
	private Stream attStatStream;
		
	/**
	 * Assume there are total K classes, M attribute, each attributes has V(M) values
	 * number of occurrence of (class,attributeIndex,a) 
	 * number of (k, a, v)
	 * Store P(Xi|Cj) for attributes from iMin to iMax
	 */
	private AttributeClassObserver[] attributeObservers;
	
	
	//private constructor based on Builder pattern
	private NBAttributeStat(Builder builder){
		this.dataset = builder.dataset;
		this.p1 = builder.p1;
		this.header = new InstancesHeader(dataset);
		this.updateFrequency = builder.updateFrequency;
	}
		
	@Override
	public boolean process(ContentEvent event) {
		
		//logger.info("begin process. id={}",this.processorId);//|!|debug
		
		if (iRange<=0)
		{
			logger.info("iRange<=0");//|!|debug
			return false; //iMax<iMin, this processor should be useless.
		}
		
		boolean bSucess = false; // if event is supported event
		
		// -------------- train or test a single instance ------------------
		if (event instanceof InstanceContentEvent)
		{
			InstanceContentEvent ice = (InstanceContentEvent)event;
			if (ice.isTraining())
			{
				// -------- Training phase -----------
				Instance inst = ice.getInstance();
				trainOnInstance(inst);

				//logger.info("instance trained.");//|!|debug
				
			}
			
			// do not process testing instances 
			bSucess = true;
		}
		
		// --------------- train multiple instances ---------------------
		if (event instanceof InstancesContentEvent)
		{
			//batch processing
			InstancesContentEvent ice = (InstancesContentEvent)event;
			if (ice.isTraining())
			{
				// -------- Training phase -----------
				Instance[] insts = ice.getInstances();
				for (int i=0;i<insts.length;i++)
				{
					Instance inst = insts[i];
					trainOnInstance(inst);
				}

				//logger.info("instances trained.");//|!|debug
			}
			
			// do not process testing instances
			
			bSucess = true;
		}
		
		//TODO: del this debug code later
		if (event.isLastEvent())
			showFinalStat();
			
		//update nbmodel through attStatStream after every F instances were trained.
		if (this.nTrainedInstancesBeforeUpdate >= this.updateFrequency)
		{
			updateModel();
			this.nTrainedInstancesBeforeUpdate=0; //restart a batch of train
		}
		return bSucess;
	}

	/**
	 * show final statistic, for debug
	 * @param event
	 */
	private void showFinalStat() {
			logger.info(" ------- NBAttributeStat(id={}) finished training",this.processorId);
			logger.info("K={},iMin={},iMax={}",this.K,this.iMin,this.iMax);
			
			//N - totally number of training instances
			logger.info("nTrain={}",this.nTrain);
			
			//Nj - number of training instances for each class Cj
			StringBuilder sb = new StringBuilder();
			for (int j=0;j<this.K;j++)
			{
				sb.append( "|"+this.nTrain_j[j]); //Nj
			}
			
			logger.info("nTrain[0~{}]={}",this.K,sb.toString());
	}

	@Override
	public void onCreate(int id) {
		// In OnCreate(id) function, need to set the range of attribute of P(Xi|Cj)
		// this processor records. iMin<=i<=iMax
		// iMin=id*( floor(m/P1) )
		// iMax=iMin+floor(m/p1)-1
		// Assume id starts from 0, m is the total number of attributes, including
		// the class attribute.
		// When id=p1-1, iMax=m-1

		this.processorId = id;
		int M = this.header.numAttributes();
		this.iMin = id*(int)(Math.floor(M/p1));
		this.iMax = this.iMin+ (int)Math.floor(M/p1) -1;
		if (id==p1-1) //the last NBAttributeStat
		{
			iMax=M-1;
		}
		this.iRange = this.iMax-this.iMin+1;
		if (iRange<=0) //iMax<iMin
			return;
		
		this.attributeObservers = new AttributeClassObserver[iRange];
		
		//Nj for each class j
		this.K = this.header.numClasses();
		this.nTrain_j = new int[this.K];
		
		for (int i=0;i<iRange;i++)
		{
			AttributeClassObserver obs=
					this.header.attribute(i).isNominal() ? newNominalClassObserver()
	                        : newNumericClassObserver();
	       this.attributeObservers[i]=obs;
		}
		
		//reset train cycle
		nTrainedInstancesBeforeUpdate = 0;
		
		logger.info("NBAttributeStat created, id={}, iMin={}, iMax={}",id,iMin,iMax);
	}

	@Override
	public Processor newProcessor(Processor p) {
		NBAttributeStat pp = (NBAttributeStat)p;
		NBAttributeStat newP = new NBAttributeStat.Builder(pp).build();
		
		//newP.trainStream = pp.trainStream;
		//newP.testStream = pp.testStream;
		newP.attStatStream = pp.attStatStream;
		
		return newP;
	}



//	public void setTrainStream(Stream trainStream) {
//		this.trainStream = trainStream;
//	}
//
//	public void setTestStream(Stream testStream) {
//		this.testStream=testStream;
//	}

	public void setAttStatStream(Stream attStatStream) {
		this.attStatStream=attStatStream;
	}

	private AttributeClassObserver newNominalClassObserver() {
		return new NominalAttributeClassObserver();
	}

	private AttributeClassObserver newNumericClassObserver() {
		return new GaussianNumericAttributeClassObserver();
	}

	
	private void trainOnInstance(Instance inst) {
        //this.nClass.addToValue((int) inst.classValue(), inst.weight());
		
		// record N, and Nj, P(Cj)=Nj/N
		int Cj = (int)inst.classValue();
		this.nTrain_j[Cj]++;
		this.nTrain++;
		this.nTrainedInstancesBeforeUpdate++;
		
        for (int ii = 0; ii < this.iRange; ii++) {
        	int i = this.iMin + ii;
            AttributeClassObserver obs = this.attributeObservers[ii];
            
        	if ( i==inst.classIndex() )
        	{
        		//special case:
        		// P(Cj) is also recorded in this processor
        		// so just record it P(Cj) in the store space of P( X[iClass] | C0 )
        		obs.observeAttributeClass(inst.classValue(), 0, inst.weight());
        		continue;
        	}
        	
              
           	// no need {{ 
//            if (obs == null) {
//                obs = inst.attribute(i).isNominal() ? newNominalClassObserver()
//                        : newNumericClassObserver();
//                this.attributeObservers[i]=obs;
//            }
        	//}} no need
            
            // error detect and ingorance
            double v = inst.value(i);
            if ( (v>=inst.attribute(i).numValues() || v<0) && inst.attribute(i).isNominal())
            {
            	//invalid data, ignore it
            	logger.error("invalid data [{}] at attribute [{}].\nRecord:[{}]",v,i,inst.toString());
            	//debugInstanceValues(inst);
            	return;//stop train this instance
            }
            else
            {
            	obs.observeAttributeClass(v, (int) inst.classValue(), inst.weight());
            }
        }
		
    }
	
	//update NBModel
	private void updateModel()
	{
		//send attribute observers to attStatStream, with iMin and iMax
		NBAttStatEvent attStatEvt = new NBAttStatEvent(this.attributeObservers, this.iMin,this.iMax);
		this.attStatStream.put(attStatEvt);
	}
	
	
	static class Builder
	{
		//required parameters
		private Instances dataset; //store the attribute information
		private int p1; //parallel number of NBAttributeStat (see NaiveBayes.nParallelAttributeStat)
		private int updateFrequency=200; // update NBModel every F instances
		
		public Builder()
		{}
		
		public Builder(NBAttributeStat p)
		{
			this.dataset = p.dataset;
			this.p1 = p.p1;
			this.updateFrequency = p.updateFrequency;
		}
		
		public Builder updateFrequency(int val){
			this.updateFrequency = val;
            return this;
		}
		
		NBAttributeStat build()
		{
			return new NBAttributeStat(this);
		}
		
		public Builder dataset(Instances val){
			this.dataset = val;
            return this;
		}
		
		public Builder p1(int val){
			this.p1 = val;
            return this;
		}
	}
}
