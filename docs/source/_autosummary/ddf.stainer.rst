ddf.stainer
===========

Stores all the stainers which are to be applied on to dataframes. All stainers should inherit from the Stainer class. Each stainer should override the transform method which dictates how the stainer will change the dataframe. The transform method returns the new dataframe, the row mapping and the column mapping. The mapping describes the positional change of indices of the original dataframe and the transformed dataframe.

.. automodule:: ddf.stainer

   
   
   

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :toctree:
   
      BinningStainer
      DateFormatStainer
      DatetimeFormatStainer
      DatetimeSplitStainer
      FTransformStainer
      InflectionStainer
      LatlongFormatStainer
      LatlongSplitStainer
      NullifyStainer
      RowDuplicateStainer
      ShuffleStainer
      Stainer
   
   

   
   
   



